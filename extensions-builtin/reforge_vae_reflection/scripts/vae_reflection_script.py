import torch
from torch import nn
import logging
import gradio as gr
from modules import scripts, script_callbacks, shared

class VAEReflectionScript(scripts.Script):
    def __init__(self):
        self.enabled = False
        self.original_padding_modes = {}

    sorting_priority = 19

    def title(self):
        return "VAE Reflection Padding"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion(open=False, label=self.title()):
            gr.HTML("<p><i>Applies reflection padding to VAE Conv2d layers for improved quality with certain VAE models.</i></p>")
            enabled = gr.Checkbox(label="Enable VAE Reflection Padding", value=self.enabled)

        enabled.change(
            lambda x: self.update_enabled(x),
            inputs=[enabled]
        )

        return [enabled]

    def update_enabled(self, value):
        self.enabled = value
        # Apply or remove reflection padding immediately
        if hasattr(shared, 'sd_model') and shared.sd_model is not None:
            if value:
                self.apply_reflection_padding()
            else:
                self.remove_reflection_padding()

    def apply_reflection_padding(self):
        """Apply reflection padding to all Conv2d layers in the current VAE model"""
        try:
            vae_model = self.get_current_vae_model()
            if vae_model is None:
                return
            
            reflection_applied = False
            for name, module in vae_model.named_modules():
                if isinstance(module, nn.Conv2d):
                    pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
                    if pad_h > 0 or pad_w > 0:
                        # Store original padding mode for restoration
                        if name not in self.original_padding_modes:
                            self.original_padding_modes[name] = module.padding_mode
                        module.padding_mode = "reflect"
                        reflection_applied = True
            
            if reflection_applied:
                print("Applied reflection padding to VAE Conv2d layers")
                
        except Exception as e:
            logging.warning(f"Failed to apply VAE reflection padding: {e}")

    def remove_reflection_padding(self):
        """Remove reflection padding and restore original padding modes"""
        try:
            vae_model = self.get_current_vae_model()
            if vae_model is None:
                return
                
            restored = False
            for name, module in vae_model.named_modules():
                if isinstance(module, nn.Conv2d) and name in self.original_padding_modes:
                    module.padding_mode = self.original_padding_modes[name]
                    restored = True
            
            if restored:
                print("Removed reflection padding from VAE Conv2d layers")
                
        except Exception as e:
            logging.warning(f"Failed to remove VAE reflection padding: {e}")

    def get_current_vae_model(self):
        """Get the current VAE model from shared.sd_model"""
        if not hasattr(shared, 'sd_model') or shared.sd_model is None:
            return None
            
        # Try forge_objects structure first
        if hasattr(shared.sd_model, 'forge_objects') and shared.sd_model.forge_objects.vae is not None:
            return shared.sd_model.forge_objects.vae.first_stage_model
        # Fallback to direct first_stage_model
        elif hasattr(shared.sd_model, 'first_stage_model'):
            return shared.sd_model.first_stage_model
            
        return None

    def process_before_every_sampling(self, p, *args, **kwargs):
        if len(args) >= 1:
            self.enabled = args[0]
        
        # Apply reflection padding if enabled
        if self.enabled:
            self.apply_reflection_padding()

def on_model_loaded(sd_model):
    """Called when a model is loaded - apply reflection if enabled"""
    # Check if there's an active script instance and if it's enabled
    for script in scripts.scripts_txt2img.scripts:
        if isinstance(script, VAEReflectionScript) and script.enabled:
            script.apply_reflection_padding()
            break

# Register the callback
script_callbacks.on_model_loaded(on_model_loaded)