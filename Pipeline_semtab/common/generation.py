from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass, field

@dataclass
class GenerationPrompt:
    messages: list
    text: str
    template_options: dict = field(default_factory=dict)

class GenerationEngine(ABC):
    @abstractmethod
    def generate(self, user_msgs, system_prompt="", **options):
        raise NotImplementedError

    def base_model(self):
        return nullcontext()

class HuggingFaceEngine(GenerationEngine):
    def __init__(self, config, token_callback=None):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.torch = torch
        self.token_callback = token_callback
        self.tokens_in = 0
        self.tokens_out = 0
        self.model_name = config.get("MODEL_NAME")
        device_id = int(config.get("DEVICE_ID", "0"))
        self.batch_size = int(config.get("LLM_BATCH_SIZE", "16"))
        max_ctx = config.get("MAX_CTX", "8192")
        self.max_ctx = None if str(max_ctx).lower() == "model" else int(max_ctx)
        self.adapter_path = (config.get("ADAPTER_PATH", "") or "").strip() or None
        self.load_in_4bit = str(config.get("LOAD_IN_4BIT", "false")).strip().lower() == "true"
        token_source = self.adapter_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(token_source)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            self.device = f"cuda:{device_id}"
            dtype_name = config.get("MODEL_DTYPE", "auto").strip().lower()
            dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
            if dtype_name == "auto":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            elif dtype_name in dtypes:
                dtype = dtypes[dtype_name]
            else:
                raise ValueError(f"Unknown model dtype: {dtype_name}")
        else:
            self.device = "cpu"
            dtype = torch.float32

        use_4bit = self.load_in_4bit and torch.cuda.is_available()
        if use_4bit:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, quantization_config=bnb_config, device_map={"": device_id})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
            self.model = self.model.to(self.device)

        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path,device_map={"": device_id} if use_4bit else None)
            if not use_4bit:
                self.model = self.model.to(self.device)

        self.model.eval()
        if self.max_ctx is None:
            self.max_ctx = getattr(self.model.config, "max_position_embeddings", 1024)
        self.has_chat_template = getattr(self.tokenizer, "chat_template", None) is not None

    def base_model(self):
        if self.adapter_path:
            return self.model.disable_adapter()
        return nullcontext()

    def format_prompt(self, system_prompt, user_msg):
        if isinstance(user_msg, GenerationPrompt):
            if not self.has_chat_template:
                return user_msg.text
            return self.tokenizer.apply_chat_template(user_msg.messages, tokenize=False,
                        add_generation_prompt=True, **user_msg.template_options)
        if not self.has_chat_template:
            return f"{system_prompt}\n\n{user_msg}\n" if system_prompt else f"{user_msg}\n"

        def apply_template(messages):
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False)
            except TypeError:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_msg})
        try:
            return apply_template(messages)
        except Exception:
            merged = f"{system_prompt}\n\n{user_msg}" if system_prompt else user_msg
            return apply_template([{"role": "user", "content": merged}])

    def generate(self, user_msgs, system_prompt="", max_new_tokens=2048, batch_size=None,do_sample=False, temperature=1.0, top_p=1.0, num_return_sequences=1, max_ctx=None):
        if batch_size is None:
            batch_size = self.batch_size
        gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample,"num_return_sequences": num_return_sequences,"pad_token_id": self.tokenizer.pad_token_id}
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        outputs = []
        for i in range(0, len(user_msgs), batch_size):
            chunk = user_msgs[i:i + batch_size]
            prompts = [self.format_prompt(system_prompt, m) for m in chunk]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True,truncation=True, max_length=self.max_ctx if max_ctx is None else max_ctx)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with self.torch.no_grad():
                gen = self.model.generate(**inputs, **gen_kwargs)
            new_tokens = gen[:, inputs["input_ids"].shape[1]:]

            if "attention_mask" in inputs:
                in_tokens = int(inputs["attention_mask"].sum().item())
            else:
                in_tokens = int(inputs["input_ids"].numel())
            in_tokens *= max(1, int(num_return_sequences))
            out_tokens = int((new_tokens != self.tokenizer.pad_token_id).sum().item())
            self.tokens_in += in_tokens
            self.tokens_out += out_tokens
            if self.token_callback is not None:
                self.token_callback(in_tokens, out_tokens)

            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend(d.strip() for d in decoded)
        return outputs

GENERATION_ENGINES = {"huggingface": HuggingFaceEngine}

def build_generation_engine(config, token_callback=None):
    name = config.get("GENERATION_ENGINE", "huggingface").strip().lower()
    if name not in GENERATION_ENGINES:
        raise ValueError(f"Unknown generation engine: {name}")
    return GENERATION_ENGINES[name](config, token_callback=token_callback)
