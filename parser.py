#############################
#    CDS403 FINAL PROJECT   # -> Parser
#############################

# Andrea Alvarez Campos G01533756


def prepare_for_env(filepath):
    print("[INFO]: Preparing map for env")
    try:
        print("[INFO]: Loading map data from file")
        with open(filepath, "r") as map_data:
            mapping = map_data.readlines()
    
    except FileNotFoundError:
        raise Exception("Specified file does not exist")
    
    for i,  row in enumerate(mapping):
        new_row = row.replace("\n", "")
        mapping[i] = new_row
    print("[DEBUG]: Map prepared for env")
    return mapping


if __name__ == "__main__":
    prepare_for_env("map_1.txt")
