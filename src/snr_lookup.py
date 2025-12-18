
def load_snr_lookup(csv_path: str = '/home/ilay.kamai/work/TalkingLatents/logs/2025-07-29/info_full.csv') -> dict:
    import csv
    import math
    snr_map = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames: return {}
            
            for row in reader:
                obsid = None
                if 'obsid' in row:
                    obsid = str(row['obsid']).strip()
                elif 'obsid.1' in row:
                    obsid = str(row['obsid.1']).strip() # Fallback or alternative
                
                if obsid and 'snrg' in row:
                    try:
                        val = float(row['snrg'])
                        if not(math.isnan(val) or math.isinf(val)):
                            snr_map[obsid] = val
                    except (ValueError, TypeError):
                        pass
    except Exception as e:
        print(f"Warning: Could not load SNR lookup: {e}")
    
    print(f"Loaded {len(snr_map)} SNR values from {csv_path}")
    return snr_map
