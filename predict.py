# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import torch
from transformers import AutoModel, AutoTokenizer
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        torch.set_grad_enabled(False)
        self.model = (
            AutoModel.from_pretrained(
                "internlm/internlm-xcomposer2-vl-1_8b",
                cache_dir="model_cache",
                trust_remote_code=True,
            )
            .cuda()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "internlm/internlm-xcomposer2-vl-1_8b", trust_remote_code=True
        )

    def predict(
        self,
        media: Path = Input(description="Input image or video file (supports MP4).", default=None),
        text: str = Input(description="Input text."),
    ) -> str:
        """Run a single prediction on the model"""
        query = f"<ImageHere>{text}" if media else text
        media_path = str(media) if media else None

        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=query, image=media_path, history=[], do_sample=False)
        
        return response