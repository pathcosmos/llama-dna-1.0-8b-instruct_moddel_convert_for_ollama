#!/usr/bin/env python3
"""
dnotitia/llama-dna-1.0-8b-instruct model convert
UV env exe - package version
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional
import argparse

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoConfig
)
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelConverter:
    def __init__(self, model_name: str, output_dir: str, use_8bit: bool = False):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.use_8bit = use_8bit
        
    def check_system_resources(self):
        """check system resource"""
        logger.info("=== check system resource ===")
        
        # GPU check
        if torch.cuda.is_available():
            logger.info(f"CUDA able  : {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            logger.info("CUDA disable")
            
        # memory check - linux
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
                for line in meminfo.split('\n'):
                    if 'MemTotal' in line:
                        total_mem = int(line.split()[1]) / 1024 / 1024  # GB
                        logger.info(f"SYSTEM RAM: {total_mem:.1f} GB")
                        break
        except:
            logger.info("failed for memory info retrieve")
            
        # check disk free space
        stat = os.statvfs('.')
        free_space = stat.f_bavail * stat.f_frsize / 1e9
        logger.info(f"AVAILABLE DISK FREE SPACE: {free_space:.1f} GB")
        
        if free_space < 30:
            logger.warning("DISK FREE SPACE IS COULD NOT BE SATISFIED(AT LEAST 30GB or more needed)")
    
    def download_model_info(self):
        """MODEL INFO PRE DOWNLOAD"""
        logger.info("=== MODEL INFO CHECK ===")
        
        try:
            # pre download setting files first
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info(f"model architecture : {config.architectures}")
            logger.info(f"words size         : {config.vocab_size}")
            if hasattr(config, 'hidden_size'):
                logger.info(f"hidden size: {config.hidden_size}")
            if hasattr(config, 'num_hidden_layers'):
                logger.info(f"num_hidden_layers: {config.num_hidden_layers}")
        except Exception as e:
            logger.error(f"failed for get model information: {e}")
            return False
        
        return True
    
    def convert_model(self):
        """execute model convert"""
        logger.info(f"=== start model convert: {self.model_name} ===")
        
        # make output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. tokenizer load
            logger.info("tokenizer download and loading...")
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True,
                cache_dir="./cache"
            )
            
            # 2. setting for model load
            model_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "trust_remote_code": True,
                "cache_dir": "./cache"
            }
            
            if self.use_8bit:
                logger.info("8bit quantumized model load...")
                try:
                    import bitsandbytes
                    model_kwargs["load_in_8bit"] = True
                except ImportError:
                    logger.warning("bitsandbytes has not installed. executtion as normal load.")
                    self.use_8bit = False
            
            if not self.use_8bit:
                logger.info("model load as normal load...")
            
            # 3. model load
            logger.info("model download and loading... (it takes time...)")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **model_kwargs
            )
            
            # 4. model save
            logger.info(f"model saving on...  {self.output_dir}")
            model.save_pretrained(
                self.output_dir,
                safe_serialization=True  # save as safetensors
            )
            
            # 5. save tokenizer
            logger.info("saving tokenizer...")
            tokenizer.save_pretrained(self.output_dir)
            
            # 6. create model info file
            model_info = {
                "original_model": self.model_name,
                "conversion_date": str(torch.tensor(1).item()),  # record present time indirectly
                "model_type": str(type(model)),
                "tokenizer_type": str(type(tokenizer)),
                "vocab_size": tokenizer.vocab_size,
                "quantized": self.use_8bit,
                "pytorch_version": torch.__version__
            }
            
            try:
                import transformers
                model_info["transformers_version"] = transformers.__version__
            except:
                pass
            
            with open(self.output_dir / "conversion_info.json", "w") as f:
                json.dump(model_info, f, indent=2, default=str)
            
            logger.info("=== convert completed! ===")
            logger.info(f"converted model located at: {self.output_dir.absolute()}")
            
            # 7. simple test
            self.test_model(model, tokenizer)
            
            return True
            
        except Exception as e:
            logger.error(f"occured error on model convert: {e}")
            return False
    
    def test_model(self, model, tokenizer):
        """coverted model test"""
        logger.info("=== model test ===")
        
        test_prompts = [
            "Hello, how are you?",
            "What is DNA?",
        ]
        
        for prompt in test_prompts:
            try:
                logger.info(f"test input: {prompt}")
                
                inputs = tokenizer(prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        max_length=100,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                logger.info(f"print out test: {response}")
                logger.info("-" * 50)
                
                break  # only test for the first
                
            except Exception as e:
                logger.error(f"error while on test: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert dnotitia/llama-dna-1.0-8b-instruct model")
    parser.add_argument(
        "--model-name", 
        default="dnotitia/llama-dna-1.0-8b-instruct",
        help="Hugging Face model name"
    )
    parser.add_argument(
        "--output-dir", 
        default="./converted_model",
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--use-8bit", 
        action="store_true",
        help="Use 8bit quantization to save memory"
    )
    parser.add_argument(
        "--check-only", 
        action="store_true",
        help="Only check system and model info, don't convert"
    )
    
    args = parser.parse_args()
    
    converter = ModelConverter(args.model_name, args.output_dir, args.use_8bit)
    
    # check system resource
    converter.check_system_resources()
    
    # check model info
    if not converter.download_model_info():
        logger.error("can not check model info.")
        sys.exit(1)
    
    if args.check_only:
        logger.info("do check then exit.")
        return
    
    # execute model convert
    if converter.convert_model():
        logger.info("model convert has succesfully completed.")
    else:
        logger.error("failed on model convert.")
        sys.exit(1)

if __name__ == "__main__":
    main()
