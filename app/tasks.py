import logging
logging.basicConfig(level=logging.DEBUG)

from celery import Celery
from app import quant_tavern_ui

# Configure Celery to use Redis as the broker and result backend.
celery_app = Celery(
    'spongequant',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task(bind=True)
def run_quantization(self, model_ids, hf_token, username,
                     gguf_sel, gguf_param,
                     gptq_sel, gptq_param,
                     exllamav2_sel, exllamav2_param,
                     awq_sel, awq_param,
                     hqq_sel, hqq_param,
                     enable_imatrix,
                     calibration_file,
                     recompute_imatrix,
                     imatrix_process_output, imatrix_verbosity, imatrix_no_ppl,
                     imatrix_chunk, imatrix_output_frequency, imatrix_save_frequency,
                     imatrix_in_files, imatrix_ngl,
                     delete_original, delete_quantized):
    full_log = ""
    # Call the generator and accumulate log output.
    for line in quant_tavern_ui(
        model_ids, hf_token, username,
        gguf_sel, gguf_param,
        gptq_sel, gptq_param,
        exllamav2_sel, exllamav2_param,
        awq_sel, awq_param,
        hqq_sel, hqq_param,
        enable_imatrix,
        calibration_file,
        recompute_imatrix,
        imatrix_process_output, imatrix_verbosity, imatrix_no_ppl,
        imatrix_chunk, imatrix_output_frequency, imatrix_save_frequency,
        imatrix_in_files, imatrix_ngl,
        delete_original, delete_quantized
    ):
        full_log += line
        # Optionally, update Celery state to report progress.
        self.update_state(state='PROGRESS', meta={'log': full_log})
    return full_log
