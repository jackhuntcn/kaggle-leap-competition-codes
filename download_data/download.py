import huggingface_hub

if __name__ == '__main__':
    retry_times = 0
    while retry_times < 1000:
        try:
            huggingface_hub.snapshot_download(
                    repo_id="LEAP/ClimSim_low-res",
                    repo_type="dataset", 
                    allow_patterns="train/*-*/*.nc", 
                    max_workers=2,
                    cache_dir=None, 
                    local_dir="."
            )
            break
        except:
            retry_times += 1
            continue
