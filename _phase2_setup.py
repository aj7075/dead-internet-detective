from huggingface_hub import HfApi, create_repo

import os
TOKEN = os.environ.get("HF_TOKEN_P2", "")  # set HF_TOKEN_P2 env var before running
api = HfApi(token=TOKEN)

try:
    create_repo('PriyanshuHF/dead-internet-detective-model-p2', token=TOKEN,
                repo_type='model', exist_ok=True, private=False)
    print('MODEL_REPO: ok')
except Exception as e:
    print('MODEL_REPO ERR:', e)

try:
    create_repo('PriyanshuHF/dead-internet-detective-trainer-p2', token=TOKEN,
                repo_type='space', space_sdk='docker', space_hardware='a10g-small',
                exist_ok=True, private=False)
    print('SPACE: ok')
except Exception as e:
    print('SPACE ERR:', e)

try:
    api.add_space_secret(repo_id='PriyanshuHF/dead-internet-detective-trainer-p2',
                         key='HF_TOKEN', value=TOKEN)
    print('SECRET HF_TOKEN: ok')
except Exception as e:
    print('SECRET ERR:', e)

api.upload_folder(folder_path=r'D:\OpenEnv_Hackathon\02_Project_File\training_space_phase2',
                  repo_id='PriyanshuHF/dead-internet-detective-trainer-p2',
                  repo_type='space',
                  commit_message='phase 2 trainer: easy+medium+hard, 20 steps, 6 ep_steps')
print('UPLOAD: ok')

try:
    api.request_space_hardware(repo_id='PriyanshuHF/dead-internet-detective-trainer-p2',
                               hardware='a10g-small')
    print('HARDWARE a10g-small: requested')
except Exception as e:
    print('HW ERR:', e)
