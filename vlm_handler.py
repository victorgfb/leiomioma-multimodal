
import torch
import torch.nn.functional as nnf

from tqdm import trange

from PIL import Image

from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM


import torch.nn as nn
from torch.utils.data import Dataset


from transformers import AutoTokenizer
from peft.optimizers import create_loraplus_optimizer

from huggingface_hub import login


import torch
import os, random
import numpy as np
import io
import base64
from transformers import set_seed

SEED = 1561312


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


set_seed(SEED)
generator = torch.Generator().manual_seed(SEED)

from pytorch_lightning import seed_everything

seed_everything(SEED, workers=True)

torch.set_float32_matmul_precision('high')


device = "cuda"

model_id = "HuggingFaceTB/SmolLM2-135M"

llm_tokenizer = AutoTokenizer.from_pretrained(model_id, device_map=device)

class MLP(nn.Module):
    def pixel_shuffle(self, x):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / self.scale_factor), embed_dim * self.scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / self.scale_factor), int(height / self.scale_factor), embed_dim * (self.scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (self.scale_factor**2)), embed_dim * (self.scale_factor**2))
        return x

    def forward(self, input):
        input_shuffle = self.pixel_shuffle(input)
        return self.model(input_shuffle)

    def __init__(self, input_size, output_size, scale_factor=2):
        super(MLP, self).__init__()

        self.scale_factor = scale_factor

        self.model = nn.Sequential(
            nn.Linear(input_size* (self.scale_factor**2), output_size), 
   
        )

import pytorch_lightning as L
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR

class PathCaptionModel(L.LightningModule):
    
    def __init__(self, mlp_shape, llm, lr = 0.005,warmup_steps = 152):
        super().__init__()

        self.llm = llm.eval().to(self.device)
        self.projection = MLP(input_size=mlp_shape[0], output_size=mlp_shape[1])
        self.projection.eval().to(self.device)

        # for param in self.projection.parameters():
        #     param.requires_grad = False

        self.warmup_steps = 0
        self.total_steps = 3000
        self.lr = lr
   

    def forward(self, input):
        image_embeds = input["image_embeds"].to(self.device)
        input_ids = input["input_ids"].to(self.device)
        attention_mask = input["attention_mask"].to(self.device)
        labels = input["labels"].to(self.device)
        
        image_proj = self.projection(image_embeds)

        token_embeds = self.llm.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat((image_proj, token_embeds), dim=1)

        image_mask = torch.ones(image_proj.shape[:2], dtype=torch.long, device=self.device)
        combined_mask = torch.cat((image_mask, attention_mask), dim=1)

        prefix_labels = torch.full(image_mask.shape, -100, dtype=torch.long, device=self.device)
        combined_labels = torch.cat((prefix_labels, labels), dim=1)


        labels = input["labels"].to(self.device)

        llm_output = self.llm.forward(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            labels=combined_labels,
            return_dict=True
        )

        return llm_output
        

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("train_loss", loss)

        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        self.log("val_loss", loss)

        return loss

    
    def configure_optimizers(self):
        projection_params = list(self.projection.parameters())
        llm_params = [p for p in self.llm.parameters() if p.requires_grad]

        # Add warmup scheduler
        optimizer = torch.optim.AdamW([
            {"params": projection_params, "lr": self.lr},  # e.g., higher lr for projection
            {"params": llm_params, "lr": self.lr}          # lower lr for LLM
        ])

        # 1. Warmup scheduler
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            return 1.0

        warmup_scheduler = LambdaLR(optimizer, lr_lambda)

        # main_scheduler = CosineAnnealingLR(
        #     optimizer, 
        #     T_max=self.total_steps - self.warmup_steps,
        #     eta_min=5e-6
        # )


        # lr_scheduler = SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_scheduler, main_scheduler],
        #     milestones=[self.warmup_steps]
        # )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warmup_scheduler,
                "interval": "step", # Update the scheduler every step
                "frequency": 1,
            },
        }




from seq_aux_dataloader_aug import siglipFinetuner
from transformers import SiglipProcessor, SiglipModel
import torch

def load_siglip_model():
    model_path = "google/siglip-base-patch16-224"
    siglip = SiglipModel.from_pretrained(
        model_path,
        device_map="cuda"
    )
    siglip_processor = SiglipProcessor.from_pretrained(model_path)

    siglip = torch.compile(siglip)
    siglip = siglipFinetuner(siglip, siglip_processor, lr = 5e-5, only_projection=False)

    checkpoint = torch.load("/run/media/victor/pessoal/mestrado/codigo/train/checkpoint/V6_phase_2/siglip-epoch4-val_loss4.12122-batch200-lr2.50e-05-v1931-seed1561312-modelo_basesiglip-epoch1-val_loss2.78238-batch100-lr5.00e-05-v1347-seed1561312-modelo_basephase_1.ckpt")
    siglip.load_state_dict(checkpoint["state_dict"])
    siglip = siglip.siglip_model
    siglip.to(device)

    return siglip


def load_llm_model():
    mlp_shape = (768,576)
    
    llm = AutoModelForCausalLM.from_pretrained(model_id,  device_map=device)

    peft_args = LoraConfig(
        lora_alpha=64,
        lora_dropout=0.1,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",

    )

    model = PathCaptionModel(mlp_shape, llm)
    model.llm = get_peft_model(model.llm, peft_args)   

    checkpoint = torch.load("/run/media/victor/pessoal/mestrado/codigo/train/checkpoint/slm-2-phase-lora-32-epoch2-val_loss2.856-batch18-lr1.00e-03-warmup-0-v2276-seed1561312-v6-llm-1-phase-llm-freese-epoch6-val_loss3.098340-batch26-v2010-seed1561312.ckpt", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to("cuda")

    return model

# Global models
VLM_MODEL = None
SIGLIP_MODEL = None
SIGLIP_PROCESSOR = None

def load_models():
    """Load models globally for FastAPI"""
    global VLM_MODEL, SIGLIP_MODEL, SIGLIP_PROCESSOR
    
    if VLM_MODEL is not None and SIGLIP_MODEL is not None and SIGLIP_PROCESSOR is not None:
        return VLM_MODEL, SIGLIP_MODEL, SIGLIP_PROCESSOR
    
    print("Loading models...")
    siglip_model = load_siglip_model()
    vlm_model = load_llm_model()
    
    # Load processor for SigLIP
    from transformers import SiglipProcessor
    siglip_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-224")
    
    # Set models to eval mode for deterministic inference
    siglip_model.eval()
    vlm_model.eval()
    vlm_model.llm.eval()
    vlm_model.projection.eval()
    
    VLM_MODEL = vlm_model
    SIGLIP_MODEL = siglip_model
    SIGLIP_PROCESSOR = siglip_processor
    
    print("Models loaded successfully!")
    return VLM_MODEL, SIGLIP_MODEL, SIGLIP_PROCESSOR

def get_description(img_path=None, img_pil=None, vlm_model=None, siglip_model=None, siglip_processor=None, max_new_tokens=100):
    """
    Get description from image
    Can accept either img_path (string) or img_pil (PIL Image)
    """
    # Use global models if not provided
    if vlm_model is None or siglip_model is None or siglip_processor is None:
        vlm_model, siglip_model, siglip_processor = load_models()
    
    # Ensure models are in eval mode
    siglip_model.eval()
    vlm_model.eval()
    vlm_model.llm.eval()
    vlm_model.projection.eval()
    
    # Load image - ensure consistent processing
    if img_pil is not None:
        img = img_pil.convert("RGB")
    elif img_path is not None:
        img = Image.open(img_path).convert("RGB")
    else:
        raise ValueError("Either img_path or img_pil must be provided")
    
    pixel_values = siglip_processor(images=img, return_tensors="pt")["pixel_values"]
    
    with torch.no_grad(): 
        image_embeds = siglip_model.vision_model(pixel_values.to(device)).last_hidden_state
    
    vlm_model.projection.to(device)
    
    image_proj = vlm_model.projection(image_embeds)

    image_mask = torch.ones(image_proj.shape[:2], dtype=torch.long, device=device)

    output = vlm_model.llm.generate(inputs_embeds=image_proj.to(dtype=torch.float16).to("cuda"), attention_mask = image_mask, 
        max_new_tokens=max_new_tokens,  # Limit the generation length
        #temperature=0.01,      # Increase randomness (higher value)
        # top_k=1,
        do_sample=False,
        eos_token_id=llm_tokenizer.eos_token_id
    )

    description = llm_tokenizer.decode(output[0])
    
    # Remove special tokens from the output
    description = description.replace("<|endoftext|>", "").strip()
    
    return description


def process_image_base64(image_base64, max_new_tokens=100):
    """
    Process an image from base64 string
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get description
        description = get_description(img_pil=image, max_new_tokens=max_new_tokens)
        
        return {
            "success": True,
            "description": description
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def handler(event):
    """
    RunPod handler function
    Expected input: {
        "input": {
            "image": "base64_encoded_image_string",
            "max_new_tokens": 100  # optional
        }
    }
    """
    try:
        input_data = event.get("input", {})
        
        # Get image (required)
        image_base64 = input_data.get("image")
        if not image_base64:
            return {
                "success": False,
                "error": "No image provided. Please provide a base64-encoded image in the 'image' field."
            }
        
        # Get optional parameters
        max_new_tokens = input_data.get("max_new_tokens", 100)
        
        # Process the image
        result = process_image_base64(
            image_base64=image_base64,
            max_new_tokens=max_new_tokens
        )
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Handler error: {str(e)}"
        }


# -------------------------
# Local server (FastAPI)
"""
Expose the handler through a local HTTP server for development/testing.

POST /infer
Body:
{
  "image": "<base64>",
  "max_new_tokens": 100
}
"""
try:
    from typing import Optional
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from pydantic import BaseModel
    import uvicorn

    class InferenceInput(BaseModel):
        image: str
        max_new_tokens: Optional[int] = 100

    app = FastAPI(title="VLM Inference Server")

    @app.get("/")
    def healthcheck():
        return {"status": "ok"}

    @app.post("/infer")
    def infer(payload: InferenceInput):
        """
        Accepts base64-encoded image
        """
        event = {
            "input": {
                "image": payload.image,
                "max_new_tokens": payload.max_new_tokens,
            }
        }

        result = handler(event)
        if not result.get("success", False):
            raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))

        return result

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001)

except ImportError:
    pass
