import os

def get_expt_name(config_file, configs_dict):
    path, base_name = os.path.split(config_file)
    base_name = os.path.splitext(base_name)[0]
    holding_dir = os.path.split(path)[1]
    return os.path.join(holding_dir, base_name)
 
def expt_summary(expt_name, configs):
    barrier_str = "#"*80
    return (barrier_str + "\n\n" + expt_name + "\n\n" + barrier_str
            + "\n\n" + configs.get("desc", "NO DESCRIPTION") + "\n\n")
    
