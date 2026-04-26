import requests, time
for name, slug, host in [
    ('P1','AryanJain7031/dead-internet-detective-trainer','aryanjain7031-dead-internet-detective-trainer'),
    ('P2','PriyanshuHF/dead-internet-detective-trainer-p2','priyanshuhf-dead-internet-detective-trainer-p2'),
]:
    r = requests.get(f'https://huggingface.co/api/spaces/{slug}/runtime', timeout=15).json()
    sha = (r.get('sha') or '')[:10]
    hw = r.get('hardware', {})
    print(f'=== {name} === STAGE: {r.get("stage")} HW: {hw} SHA: {sha}')
    try:
        d = requests.get(f'https://{host}.hf.space/status', timeout=10).json()
        print(f'  PHASE: {d.get("phase")} STEP: {d.get("step")}/{d.get("total")} REWARD: {d.get("mean_reward")}')
        for l in d.get('log', [])[-8:]:
            print('   ', l)
    except Exception as e:
        print('  STATUS ERR:', e)
print('NOW:', time.strftime('%H:%M:%S'))
