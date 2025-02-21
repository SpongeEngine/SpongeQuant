#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpongeQuant

Inspired by text-generation-webui and the original Colab AutoQuant notebook,
this script provides a web-based UI for quantizing a model downloaded from Hugging Face.
It now features dynamic default parameters for each quantization method along with
the ability to select exactly which methods you want to run.
It also can compute an imatrix file (using llama-imatrix) from a provided calibration dataset,
so that users don’t have to manually run a separate tool.
"""

import os
import subprocess
import time
import sys
import json
import gradio as gr
import shutil
from huggingface_hub import snapshot_download, HfApi, ModelCard, create_repo

from format_quant_type import format_quant_type
from generate_custom_model_card import generate_custom_model_card

# ---------------------------
# Global Default Parameters
# ---------------------------
DEFAULT_IMATRIX_FILE = os.path.join("gguf", "imatrix.dat")
DEFAULT_CALIBRATION_FILE = os.path.join("gguf", "calibration_datav3.txt")

DEFAULT_PARAMS = {
    "GGUF": "IQ1_S, IQ1_M, TQ1_0, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, TQ2_0, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ4_XS, IQ4_NL, Q2_K_S, Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_0, Q4_1, Q4_K_M, Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K",
    "GPTQ": "4, 128, 0.1",
    "ExLlamaV2": "4.5",
    "AWQ": "4, 128, GEMM, True",
    "HQQ": "2, 128"
}

# ---------------------------
# Helper: Patch config file on disk
# ---------------------------
def patch_model_config(model_dir):
    """
    Modify the config.json file in model_dir to override the rope_scaling field,
    leaving only the two required keys: 'type' and 'factor'. Also remove extraneous keys
    that may confuse AWQ (e.g., 'low_freq_factor', 'high_freq_factor', 'original_max_position_embeddings', 'rope_type').
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"[WARN] No config.json found in {model_dir}")
        return
    try:
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        if "rope_scaling" in config_dict and isinstance(config_dict["rope_scaling"], dict):
            factor = config_dict["rope_scaling"].get("factor", 1.0)
            config_dict["rope_scaling"] = {"type": "linear", "factor": factor}
        for key in ["low_freq_factor", "high_freq_factor", "original_max_position_embeddings", "rope_type"]:
            if key in config_dict:
                del config_dict[key]
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
        print(f"[INFO] Patched config in {config_path}")
    except Exception as e:
        print(f"[ERROR] Failed to patch config file: {e}")

# ---------------------------
# Helper: Run a shell command and stream output
# ---------------------------
def run_command(command: str):
    """Run a shell command and yield its output line by line."""
    yield f"[DEBUG] Executing command: {command}\n"
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='ISO-8859-1'
    )
    while True:
        line = process.stdout.readline()
        if line:
            yield line
        elif process.poll() is not None:
            break
    remaining = process.stdout.read()
    if remaining:
        yield remaining
    if process.returncode not in (None, 0):
        yield f"[ERROR] Command returned non-zero exit code: {process.returncode}\n"

# ---------------------------
# Helper: Compute imatrix file
# ---------------------------
def compute_imatrix_file(model_file: str, calibration_file: str, imatrix_output: str,
                         process_output: bool, verbosity: int, no_ppl: bool,
                         chunk: int, output_frequency: int, save_frequency: int,
                         in_files: list, ngl: int):
    """
    Compute the importance matrix file using the llama-imatrix tool.
    All optional parameters for imatrix are provided.
    """
    yield f"[INFO] Computing imatrix file for model {model_file} using calibration data from {calibration_file}\n"
    cmd_parts = [
        "llama-imatrix",
        "-m", model_file,
        "-f", calibration_file,
        "-o", imatrix_output,
        "--chunk", str(chunk),
        "-ngl", str(ngl),
        "--output-frequency", str(output_frequency),
        "--save-frequency", str(save_frequency),
        "--verbosity", str(verbosity)
    ]

    if process_output:
        cmd_parts.append("--process-output")
    if no_ppl:
        cmd_parts.append("--no-ppl")
    for in_file in in_files:
        cmd_parts.extend(["--in-file", in_file])
    cmd = build_llama_cmd(*cmd_parts)
    yield f"[INFO] Running imatrix command:\n  {cmd}\n"
    for line in run_command(cmd):
        yield line

# ---------------------------
# Utility functions
# ---------------------------
def is_model_fully_downloaded(model_id: str, target_dir: str, hf_token: str) -> bool:
    """
    Checks if all expected files for a Hugging Face model exist in the local directory.
    """
    try:
        api = HfApi()
        model_files = api.list_repo_files(repo_id=model_id, token=hf_token)
        for file in model_files:
            file_path = os.path.join(target_dir, file)
            if not os.path.exists(file_path):
                print(f"[DEBUG] Missing file: {file_path}")
                return False
        return True
    except Exception as e:
        print(f"[ERROR] Error checking model files: {e}")
        return False

def download_model(model_id: str, hf_token: str):
    """Download model from Hugging Face only if not already fully downloaded."""
    model_name = model_id.split("/")[-1]
    target_dir = os.path.join("models", model_name)
    
    yield "=== Downloading Model ===\n"
    yield f"[INFO] Model ID: {model_id}\n"
    yield f"[INFO] Target directory: {target_dir}\n"
    
    if os.path.exists(target_dir):
        if is_model_fully_downloaded(model_id, target_dir, hf_token):
            yield f"[INFO] Model {model_id} is already fully downloaded at {target_dir}. Skipping download.\n"
            return
        else:
            yield "[WARN] Model directory exists but files appear incomplete. Re-downloading...\n"
    else:
        yield "[INFO] Model directory does not exist. Starting download...\n"

    try:
        yield f"[INFO] Downloading model {model_id}...\n"
        model_path = snapshot_download(
            repo_id=model_id,
            token=hf_token,
            local_dir=target_dir,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
            resume_download=True
        )
        patch_model_config(target_dir)
        yield f"[INFO] Model downloaded and patched at: {model_path}\n"
    except Exception as e:
        yield f"[ERROR] Error downloading model: {e}\n"

# ---------------------------
# New Helper: Upload with Retry (no backup copy)
# ---------------------------
def upload_quant_retry(model_id, base_model_name, quantization_type, save_folder, hf_token, username, max_retries=3, **kwargs):
    """
    Retry the upload process for the quantized model.
    If the upload succeeds, create a marker file "upload_success.txt" in the folder.
    Yields log messages.
    """
    repo_id = f"{username}/{base_model_name}-{format_quant_type(quantization_type)}"
    success = False
    for attempt in range(1, max_retries+1):
        yield f"[INFO] Upload attempt {attempt} for repository {repo_id}\n"
        log = upload_quant(model_id, base_model_name, quantization_type, save_folder, hf_token, username, **kwargs)
        yield log
        if "[ERROR]" not in log:
            success = True
            yield f"[INFO] Upload succeeded on attempt {attempt}.\n"
            # Create a marker file to indicate successful upload.
            marker_path = os.path.join(save_folder, "upload_success.txt")
            try:
                with open(marker_path, "w") as f:
                    f.write("Upload succeeded.\n")
                yield f"[INFO] Created marker file {marker_path}.\n"
            except Exception as e:
                yield f"[WARN] Failed to create marker file: {e}\n"
            break
        else:
            if attempt < max_retries:
                yield "[WARN] Upload failed. Retrying after 5 seconds...\n"
                time.sleep(5)
            else:
                yield "[ERROR] Maximum upload attempts reached. Upload failed.\n"
    # Note: The caller should check for the marker file to decide cleanup.
    return

def upload_quant(model_id, base_model_name, quantization_type, save_folder, hf_token, username, **kwargs):
    repo_id = f"{username}/{base_model_name}-{format_quant_type(quantization_type)}"
    log = f"[INFO] Preparing to upload quantized model to repo: {repo_id}\n"
    
    try:
        card_content = generate_custom_model_card(model_id, base_model_name, quantization_type, username, save_folder)
        card_path = os.path.join(save_folder, "README.md")
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(card_content)
        log += f"[INFO] Created custom model card for {repo_id} at {card_path}\n"
    except Exception as e:
        log += f"[ERROR] Error creating custom model card: {e}\n"
    
    try:
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=hf_token)
        log += f"[INFO] Repo {repo_id} is ready. Uploading folder {save_folder}...\n"
        api = HfApi()
        api.upload_folder(
            folder_path=save_folder,
            repo_id=repo_id,
            ignore_patterns=["*.bf16.gguf"],
            token=hf_token
        )
        log += f"[INFO] Uploaded quantized model to {repo_id}\n"
    except Exception as e:
        log += f"[ERROR] Error uploading model: {e}\n"
    
    return log

def build_llama_cmd(script_name: str, *args):
    """
    Build command with the appropriate path (either Python script or compiled executable).
    """
    if script_name.endswith(".py"):
        script_path = os.path.join("llama_cpp", script_name)
        return f'"{sys.executable}" "{script_path}" ' + " ".join(str(arg) for arg in args)
    else:
        exec_path = os.path.join("llama_cpp", "build", "bin", script_name)
        return f'"{exec_path}" ' + " ".join(str(arg) for arg in args)

def quantize_gguf(model_id: str, additional_param: str, hf_token: str, username: str,
                  use_imatrix: bool, calibration_file: str, recompute_imatrix: bool,
                  imatrix_process_output: bool, imatrix_verbosity: int, imatrix_no_ppl: bool,
                  imatrix_chunk: int, imatrix_output_frequency: int, imatrix_save_frequency: int,
                  imatrix_in_files: str, imatrix_ngl: int, delete_quantized: bool):
    base_model_name = model_id.split("/")[-1].strip()
    model_dir = os.path.join("models", base_model_name)
    # Use a single base folder for all GGUF outputs.
    base_save_folder = os.path.join("quantized_models", f"{base_model_name}-GGUF")
    os.makedirs(base_save_folder, exist_ok=True)
    out_file = os.path.join(base_save_folder, f"{base_model_name.lower()}.bf16.gguf")
    yield f"=== GGUF Quantization for {base_model_name} ===\n"
    yield f"[INFO] Expected base conversion output file: {out_file}\n"
    
    if use_imatrix:
        imatrix_file = os.path.join(base_save_folder, f"{base_model_name}.imatrix.dat")
    
    # Convert the model to bf16.gguf if not already done.
    if not os.path.exists(out_file):
        cmd = build_llama_cmd("convert_hf_to_gguf.py", model_dir, "--outtype", "bf16", "--outfile", out_file)
        yield f"[INFO] Running conversion command:\n  {cmd}\n"
        for line in run_command(cmd):
            yield line
    else:
        yield f"[INFO] File {out_file} already exists. Skipping conversion.\n"
    
    # Compute the imatrix file if enabled.
    if use_imatrix:
        in_files_list = [s.strip() for s in imatrix_in_files.split(",") if s.strip()] if imatrix_in_files.strip() else []
        if recompute_imatrix or not os.path.exists(imatrix_file):
            for line in compute_imatrix_file(out_file, calibration_file, imatrix_file,
                                             process_output=imatrix_process_output,
                                             verbosity=imatrix_verbosity,
                                             no_ppl=imatrix_no_ppl,
                                             chunk=imatrix_chunk,
                                             output_frequency=imatrix_output_frequency,
                                             save_frequency=imatrix_save_frequency,
                                             in_files=in_files_list,
                                             ngl=imatrix_ngl):
                yield line

    # Use a consistent repository type for all uploads.
    repo_quant_type = "i1-GGUF" if use_imatrix else "GGUF"

    # Process each quantization method sequentially.
    quant_methods = additional_param.replace(" ", "").split(",")
    for method in quant_methods:
        method_str = method.strip().upper()
        if method_str.startswith("IQ") and not use_imatrix:
            yield f"[WARN] Skipping {method_str} quantization because imatrix is not enabled.\n"
            continue

        # Build a unique output filename for this quantization method.
        if use_imatrix:
            quantized_output = os.path.join(base_save_folder, f"{base_model_name.lower()}-i1-{method_str}.gguf")
            cmd = build_llama_cmd("llama-quantize", "--imatrix", imatrix_file, out_file, quantized_output, method_str)
        else:
            quantized_output = os.path.join(base_save_folder, f"{base_model_name.lower()}-{method_str}.gguf")
            cmd = build_llama_cmd("llama-quantize", out_file, quantized_output, method_str)
        
        yield f"[INFO] Quantizing with method '{method_str}':\n  {cmd}\n"
        for line in run_command(cmd):
            yield line

        # Sequentially upload this quantized file to the unified repository.
        yield f"[INFO] Uploading quantized file {quantized_output} to repo '{repo_quant_type}'...\n"
        upload_logs = upload_quant_retry(model_id, base_model_name, repo_quant_type, base_save_folder, hf_token, username)
        for line in upload_logs:
            yield line

        # Delete the quantized file if deletion is enabled.
        if delete_quantized and os.path.exists(quantized_output):
            try:
                os.remove(quantized_output)
                yield f"[INFO] Deleted quantized file {quantized_output} after upload.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete quantized file {quantized_output}: {e}\n"

    # Optionally, delete the base conversion file and imatrix file if deletion is enabled.
    if delete_quantized:
        if os.path.exists(out_file):
            try:
                os.remove(out_file)
                yield f"[INFO] Deleted base conversion file {out_file} after all uploads.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete base conversion file {out_file}: {e}\n"
        if use_imatrix and os.path.exists(imatrix_file):
            try:
                os.remove(imatrix_file)
                yield f"[INFO] Deleted imatrix file {imatrix_file} after all uploads.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete imatrix file {imatrix_file}: {e}\n"

def quantize_gptq(model_id: str, additional_param: str, hf_token: str, username: str, delete_quantized: bool):
    try:
        from transformers import AutoTokenizer, AutoConfig, GPTQConfig, AutoModelForCausalLM
    except ImportError:
        yield "[ERROR] GPTQ quantization requires the 'transformers' package.\n"
        return

    yield "=== GPTQ Quantization ===\n"
    defaults = [4, 128, 0.1]
    if additional_param.strip():
        parts = [p.strip() for p in additional_param.split(",")]
        if len(parts) >= 3:
            bits = int(parts[0])
            group_size = int(parts[1])
            damp_percent = float(parts[2])
        else:
            bits, group_size, damp_percent = defaults
            yield f"[WARN] Insufficient GPTQ parameters provided. Using defaults: {defaults}\n"
    else:
        bits, group_size, damp_percent = defaults

    yield f"[INFO] Using GPTQ parameters: bits={bits}, group_size={group_size}, damp_percent={damp_percent}\n"

    base_model_name = model_id.split("/")[-1]
    local_dir = os.path.join("models", base_model_name)
    yield "[INFO] Loading tokenizer from local model directory...\n"
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    yield "[INFO] Loading patched configuration from local model directory...\n"
    patch_model_config(local_dir)
    config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    yield "[DEBUG] Patched config rope_scaling: " + str(config.rope_scaling) + "\n"
    yield "[INFO] Initializing GPTQ configuration...\n"
    quantization_config = GPTQConfig(
        bits=bits,
        dataset="c4",
        tokenizer=tokenizer,
        damp_percent=damp_percent,
        rope_scaling=config.rope_scaling
    )
    yield "[INFO] Loading model with integrated GPTQ configuration...\n"
    model = AutoModelForCausalLM.from_pretrained(
        local_dir,
        config=config,
        device_map="auto",
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    yield "[DEBUG] Loaded model config rope_scaling: " + str(model.config.rope_scaling) + "\n"
    model.config.rope_scaling = config.rope_scaling
    yield "[DEBUG] After override, model config rope_scaling: " + str(model.config.rope_scaling) + "\n"
    
    save_folder = os.path.join("quantized_models", f"{base_model_name}-GPTQ")
    yield f"[INFO] Saving quantized model to {save_folder}...\n"
    model.save_pretrained(save_folder, use_safetensors=True)
    tokenizer.save_pretrained(save_folder)
    yield "[INFO] GPTQ quantization completed.\n"
    # Use retry upload helper.
    upload_logs = upload_quant_retry(model_id, base_model_name, "GPTQ", save_folder, hf_token, username)
    for line in upload_logs:
        yield line

    if delete_quantized:
        if os.path.exists(save_folder):
            try:
                shutil.rmtree(save_folder)
                yield f"[INFO] Deleted quantized model folder {save_folder} after upload.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete quantized folder {save_folder}: {e}\n"

def quantize_exllamav2(model_id: str, additional_param: str, hf_token: str, username: str, delete_quantized: bool):
    try:
        yield "=== ExLlamaV2 Quantization ===\n"
        bpw = float(additional_param) if additional_param.strip() else 4.5
        base_model_name = model_id.split("/")[-1]
        model_dir = os.path.join("models", base_model_name)
        save_folder = os.path.join("quantized_models", f"{base_model_name}-EXL2")
        cmd = f'"{sys.executable}" "/app/exllamav2/convert.py" -i {model_dir} -o {save_folder} -b {bpw}'
        yield f"[INFO] Running ExLlamaV2 command:\n  {cmd}\n"
        for line in run_command(cmd):
            yield line
        yield "[INFO] ExLlamaV2 quantization completed.\n"
        upload_logs = upload_quant_retry(model_id, base_model_name, "exl2", save_folder, hf_token, username, bpw=bpw)
        for line in upload_logs:
            yield line
    except Exception as e:
        yield f"[ERROR] Error during ExLlamaV2 quantization: {e}\n"

    if delete_quantized:
        if os.path.exists(save_folder):
            try:
                shutil.rmtree(save_folder)
                yield f"[INFO] Deleted quantized model folder {save_folder} after upload.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete quantized folder {save_folder}: {e}\n"

def quantize_awq(model_id: str, additional_param: str, hf_token: str, username: str, delete_quantized: bool):
    try:
        yield "=== AWQ Quantization ===\n"
        defaults = [4, 128, "GEMM", True]
        if additional_param.strip():
            parts = [p.strip() for p in additional_param.split(",")]
            if len(parts) >= 4:
                bits = int(parts[0])
                group_size = int(parts[1])
                version = parts[2]
                zero_point = parts[3].lower() in ["true", "1", "yes"]
            else:
                bits, group_size, version, zero_point = defaults
                yield f"[WARN] Insufficient AWQ parameters provided. Using defaults: {defaults}\n"
        else:
            bits, group_size, version, zero_point = defaults

        yield f"[INFO] Using AWQ parameters: bits={bits}, group_size={group_size}, version={version}, zero_point={zero_point}\n"
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer, AutoConfig

        quant_config = {
            "w_bit": bits,
            "q_group_size": group_size,
            "version": version,
            "zero_point": zero_point
        }

        base_model_name = model_id.split("/")[-1]
        save_folder = os.path.join("quantized_models", f"{base_model_name}-AWQ")
        local_dir = os.path.join("models", base_model_name)
        yield "[INFO] Loading model and tokenizer for AWQ from local directory...\n"
        patch_model_config(local_dir)
        config = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
        if isinstance(config.rope_scaling, dict):
            config.rope_scaling = {"type": "linear", "factor": 1.0}
            yield f"[DEBUG] Overridden config rope_scaling: {config.rope_scaling}\n"
        model = AutoAWQForCausalLM.from_pretrained(local_dir, config=config, safetensors=True, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
        
        yield "[INFO] Quantizing model using AWQ...\n"
        model.quantize(tokenizer, quant_config=quant_config)
        model.save_quantized(save_folder)
        tokenizer.save_pretrained(save_folder)
        yield f"[INFO] AWQ quantization completed. Saved to {save_folder}\n"
        upload_logs = upload_quant_retry(model_id, base_model_name, "AWQ", save_folder, hf_token, username)
        for line in upload_logs:
            yield line
    except Exception as e:
        yield f"[ERROR] Error during AWQ quantization: {e}\n"

    if delete_quantized:
        if os.path.exists(save_folder):
            try:
                shutil.rmtree(save_folder)
                yield f"[INFO] Deleted quantized model folder {save_folder} after upload.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete quantized folder {save_folder}: {e}\n"

def quantize_hqq(model_id: str, additional_param: str, hf_token: str, username: str, delete_quantized: bool):
    try:
        yield "=== HQQ Quantization ===\n"
        defaults = [2, 128]
        if additional_param.strip():
            parts = [p.strip() for p in additional_param.split(",")]
            if len(parts) >= 2:
                bits = int(parts[0])
                group_size = int(parts[1])
            else:
                bits, group_size = defaults
                yield f"[WARN] Insufficient HQQ parameters provided. Using defaults: {defaults}\n"
        else:
            bits, group_size = defaults

        yield f"[INFO] Using HQQ parameters: bits={bits}, group_size={group_size}\n"
        from hqq.engine.hf import HQQModelForCausalLM, AutoTokenizer
        from hqq.core.quantize import BaseQuantizeConfig
        quant_config = BaseQuantizeConfig(nbits=bits, group_size=group_size)
        base_model_name = model_id.split("/")[-1]
        save_folder = os.path.join("quantized_models", f"{base_model_name}-HQQ")
        yield "[INFO] Downloading HQQ model and tokenizer...\n"
        model = HQQModelForCausalLM.from_pretrained(
            model_id, cache_dir=".", attn_implementation="flash_attention_2"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        yield "[INFO] Quantizing model using HQQ...\n"
        model.quantize_model(quant_config=quant_config, device='cuda')
        yield f"[INFO] HQQ quantization completed. Saving to {save_folder}...\n"
        model.save_quantized(save_folder)
        tokenizer.save_pretrained(save_folder)
        upload_logs = upload_quant_retry(model_id, base_model_name, "HQQ", save_folder, hf_token, username)
        for line in upload_logs:
            yield line
    except Exception as e:
        yield f"[ERROR] Error during HQQ quantization: {e}\n"

    if delete_quantized:
        if os.path.exists(save_folder):
            try:
                shutil.rmtree(save_folder)
                yield f"[INFO] Deleted quantized model folder {save_folder} after upload.\n"
            except Exception as e:
                yield f"[ERROR] Failed to delete quantized folder {save_folder}: {e}\n"

# ---------------------------
# Main Orchestration Function
# ---------------------------
def quant_tavern_ui(model_ids: str, hf_token: str, username: str,
                    gguf_sel: bool, gguf_param: str,
                    gptq_sel: bool, gptq_param: str,
                    exllamav2_sel: bool, exllamav2_param: str,
                    awq_sel: bool, awq_param: str,
                    hqq_sel: bool, hqq_param: str,
                    enable_imatrix: bool,
                    calibration_file: str,
                    recompute_imatrix: bool,
                    imatrix_process_output: bool, imatrix_verbosity: int, imatrix_no_ppl: bool,
                    imatrix_chunk: int, imatrix_output_frequency: int, imatrix_save_frequency: int,
                    imatrix_in_files: str, imatrix_ngl: int,
                    delete_original: bool, delete_quantized: bool):
    full_log = "=== Starting SpongeQuant Quantization Process ===\n"
    # Split the input into a list of model IDs (one per non-empty line)
    model_list = [m.strip() for m in model_ids.splitlines() if m.strip()]
    
    # Dictionary to store per-model upload status (True if all methods succeeded)
    model_upload_success = {}
    
    for model_id in model_list:
        model_success = True  # assume true; set to False if any method fails upload
        full_log += f"\n=== Processing model: {model_id} ===\n"
        for line in download_model(model_id, hf_token):
            full_log += line
            yield full_log

        selected_methods = []
        if gguf_sel:
            selected_methods.append(("GGUF", gguf_param))
        if gptq_sel:
            selected_methods.append(("GPTQ", gptq_param))
        if exllamav2_sel:
            selected_methods.append(("ExLlamaV2", exllamav2_param))
        if awq_sel:
            selected_methods.append(("AWQ", awq_param))
        if hqq_sel:
            selected_methods.append(("HQQ", hqq_param))
        
        if not selected_methods:
            full_log += "[ERROR] No quantization method selected for model. Please select at least one method.\n"
            yield full_log
            continue

        for method, param in selected_methods:
            full_log += f"[INFO] Running {method} quantization...\n"
            yield full_log
            if method == "GGUF":
                for line in quantize_gguf(model_id, gguf_param, hf_token, username,
                                          enable_imatrix, calibration_file, recompute_imatrix,
                                          imatrix_process_output, imatrix_verbosity, imatrix_no_ppl,
                                          imatrix_chunk, imatrix_output_frequency, imatrix_save_frequency,
                                          imatrix_in_files, imatrix_ngl, delete_quantized):
                    full_log += line
                    yield full_log
            elif method == "GPTQ":
                for line in quantize_gptq(model_id, param, hf_token, username, delete_quantized):
                    full_log += line
                    yield full_log
            elif method == "ExLlamaV2":
                for line in quantize_exllamav2(model_id, param, hf_token, username, delete_quantized):
                    full_log += line
                    yield full_log
            elif method == "AWQ":
                for line in quantize_awq(model_id, param, hf_token, username, delete_quantized):
                    full_log += line
                    yield full_log
            elif method == "HQQ":
                for line in quantize_hqq(model_id, param, hf_token, username, delete_quantized):
                    full_log += line
                    yield full_log
            else:
                full_log += f"[ERROR] Unknown quantization method: {method}\n"
                yield full_log

        # After processing all methods for the current model, perform cleanup of the original folder if all quantized outputs succeeded.
        base_model_name = model_id.split("/")[-1]
        quant_dir = os.path.join("quantized_models")
        model_quant_folders = [os.path.join(quant_dir, folder)
                               for folder in os.listdir(quant_dir)
                               if folder.startswith(base_model_name + "-")]
        all_success = True
        for folder in model_quant_folders:
            marker = os.path.join(folder, "upload_success.txt")
            if not os.path.exists(marker):
                all_success = False
                full_log += f"[WARN] Quantized folder '{folder}' did not mark a successful upload. It will not be auto-deleted.\n"
        if delete_original:
            original_path = os.path.join("models", base_model_name)
            if os.path.exists(original_path):
                if all_success:
                    try:
                        shutil.rmtree(original_path)
                        full_log += f"[INFO] Deleted original model folder: {original_path}\n"
                    except Exception as e:
                        full_log += f"[ERROR] Could not delete original model folder {original_path}: {e}\n"
                else:
                    full_log += f"[INFO] Skipping deletion of original folder {original_path} because not all quantizations uploaded successfully.\n"
        # Final cleanup for quantized outputs is not needed when delete_quantized is True (as deletion occurs sequentially).
        if not delete_quantized:
            for folder in model_quant_folders:
                marker = os.path.join(folder, "upload_success.txt")
                if os.path.exists(marker):
                    try:
                        shutil.rmtree(folder)
                        full_log += f"[INFO] Deleted quantized model folder: {folder}\n"
                    except Exception as e:
                        full_log += f"[ERROR] Could not delete quantized folder {folder}: {e}\n"
                else:
                    full_log += f"[INFO] Retaining quantized folder {folder} for manual intervention (upload did not succeed).\n"
        yield full_log

    full_log += "\n=== Quantization Process Completed ===\n"
    yield full_log

# ---------------------------
# Build the UI using Gradio
# ---------------------------
with gr.Blocks(title="SpongeQuant") as iface:
    gr.Markdown("# SpongeQuant")
    with gr.Row():
        model_ids_input = gr.Textbox(label="Model IDs (one per line)", 
                                     value="mlabonne/Meta-Llama-3.1-8B-Instruct-abliterated", 
                                     lines=3)
        hf_token_input = gr.Textbox(label="Hugging Face Token", type="password")
        username_input = gr.Textbox(label="Hugging Face Username", value="SpongeEngine")
    gr.Markdown("### Select Quantization Methods")
    with gr.Accordion("Quantization Methods", open=True):
        with gr.Row():
            gguf_checkbox = gr.Checkbox(label="GGUF", value=True,
                                        info="Enable GGUF quantization (supports optional imatrix calibration).")
            gguf_param = gr.Textbox(label="GGUF Additional Parameter", value=DEFAULT_PARAMS["GGUF"])
        with gr.Row():
            gptq_checkbox = gr.Checkbox(label="GPTQ", value=False,
                                        info="Enable GPTQ quantization (requires transformers).")
            gptq_param = gr.Textbox(label="GPTQ Additional Parameter", value=DEFAULT_PARAMS["GPTQ"])
        with gr.Row():
            exllamav2_checkbox = gr.Checkbox(label="ExLlamaV2", value=False,
                                             info="Enable ExLlamaV2 quantization (runs conversion script).")
            exllamav2_param = gr.Textbox(label="ExLlamaV2 Additional Parameter", value=DEFAULT_PARAMS["ExLlamaV2"])
        with gr.Row():
            awq_checkbox = gr.Checkbox(label="AWQ", value=False,
                                       info="Enable AWQ quantization (requires AWQ package).")
            awq_param = gr.Textbox(label="AWQ Additional Parameter", value=DEFAULT_PARAMS["AWQ"])
        with gr.Row():
            hqq_checkbox = gr.Checkbox(label="HQQ", value=False,
                                       info="Enable HQQ quantization (requires HQQ package).")
            hqq_param = gr.Textbox(label="HQQ Additional Parameter", value=DEFAULT_PARAMS["HQQ"])
    gr.Markdown("### Imatrix Advanced Parameters")
    with gr.Row():
        imatrix_process_output_checkbox = gr.Checkbox(label="Process output.weight", value=False,
                                                      info="If enabled, processes the output weights during imatrix computation.")
        imatrix_verbosity_input = gr.Number(label="Verbosity", value=1, precision=0)
        imatrix_no_ppl_checkbox = gr.Checkbox(label="Disable PPL", value=False,
                                              info="If enabled, disables perplexity calculation in imatrix computation.")
    with gr.Row():
        imatrix_chunk_input = gr.Number(label="Chunk size", value=64, precision=0)
        imatrix_output_freq_input = gr.Number(label="Output Frequency", value=10, precision=0)
        imatrix_save_freq_input = gr.Number(label="Save Frequency", value=0, precision=0)
    with gr.Row():
        imatrix_in_files_input = gr.Textbox(label="Additional in-files (comma-separated)", value="")
        imatrix_ngl_input = gr.Number(label="GPU offload (-ngl)", value=80, precision=0)
    gr.Markdown("### Imatrix Calibration (GGUF Only)")
    with gr.Row():
        enable_imatrix_checkbox = gr.Checkbox(label="Enable Imatrix", value=True,
                                              info="Enable imatrix computation for GGUF quantization (requires calibration file).")
    with gr.Row():
        calibration_file_input = gr.Textbox(label="Calibration Data File Path", value=DEFAULT_CALIBRATION_FILE)
        recompute_imatrix_checkbox = gr.Checkbox(label="Compute Imatrix", value=True)
    gr.Markdown("### Cleanup Options")
    with gr.Row():
        delete_original_checkbox = gr.Checkbox(
            label="Delete Original Model after quantization", 
            value=True,
            info="If enabled, the original model folder will be deleted after all quantization uploads have completed successfully."
        )
        delete_quantized_checkbox = gr.Checkbox(
            label="Delete Quantization Output after upload", 
            value=True,
            info="When enabled, each quantized output is deleted immediately after a successful upload, freeing disk space."
        )
    with gr.Row():
        run_button = gr.Button("Run Quantization")
    quant_output = gr.Textbox(label="Output Log", interactive=False, lines=20)
    
    run_button.click(
        fn=quant_tavern_ui, 
        inputs=[
            model_ids_input, hf_token_input, username_input,
            gguf_checkbox, gguf_param,
            gptq_checkbox, gptq_param,
            exllamav2_checkbox, exllamav2_param,
            awq_checkbox, awq_param,
            hqq_checkbox, hqq_param,
            enable_imatrix_checkbox,
            calibration_file_input, recompute_imatrix_checkbox,
            imatrix_process_output_checkbox, imatrix_verbosity_input, imatrix_no_ppl_checkbox,
            imatrix_chunk_input, imatrix_output_freq_input, imatrix_save_freq_input,
            imatrix_in_files_input, imatrix_ngl_input,
            delete_original_checkbox, delete_quantized_checkbox
        ], 
        outputs=quant_output
    )

if __name__ == "__main__":
    iface.queue()
    iface.launch(server_name="0.0.0.0", server_port=7860)