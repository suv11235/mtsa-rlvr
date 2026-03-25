import json
import re

def create_biosecurity_dataset():
    files_to_read = [
        "/Users/suvajitmajumder/mtsa-rlvr/MTSA/datasets/attack_target/bio_attack_target.json",
        "/Users/suvajitmajumder/mtsa-rlvr/MTSA/datasets/attack_target/train_attack_target_labels.json"
    ]
    output_file = "/Users/suvajitmajumder/mtsa-rlvr/MTSA/datasets/attack_target/biosecurity_goals.json"

    all_data = []
    for f_path in files_to_read:
        with open(f_path, 'r') as f:
            all_data.extend(json.load(f))

    # Keywords for inclusion (Biosecurity + pathogens + controlled substances)
    include_regex = re.compile(
        r"pathogen|botulinum|anthrax|plague|smallpox|ricin|ebola|bioweapon|biological|laboratory|culture|bacteria|toxin|"
        r"cyanide|sarin|nerve agent|chemical weapon|bsl-4|bsl-3|synthesis|organism|biohazard|biosecurity|disease|"
        r"epidemic|pandemic|outbreak|pathogenic|tetrodotoxin|anthrax|ebola|marburg|h5n1|mers|sars|virus|biological warfare|"
        r"isolation|identification|characterization|cultivation|fermentation|genetically modified|enhancement|virulence|"
        r"transmissibility|containment|decontamination|sterilization|autoclave|centrifugation|lyophilization", 
        re.IGNORECASE
    )
    
    # Keywords for exclusion (Cyber and unrelated misinformation)
    exclude_regex = re.compile(
        r"computer|software|network|cyber|internet|program|code|files|data|antivirus|ransomware|hacking into|database|"
        r"website|app|bypass|firewall|server|5G|racism|minority|politics|election|stock market|financial|insider trading|"
        r"scam|fraud|money laundering|marijuana|heroin|methamphetamine|fentanyl|cocaine|drug|anorexia|bulimia|suicide|"
        r"identity theft|credit card|pirated content|pirate software|copyrighted material|illegal download", 
        re.IGNORECASE
    )

    filtered_data = []
    seen_goals = set()
    for entry in all_data:
        goal = entry.get("goal", "")
        if include_regex.search(goal) and not exclude_regex.search(goal):
            if goal not in seen_goals:
                filtered_data.append({"goal": goal})
                seen_goals.add(goal)

    print(f"Total entries processed: {len(all_data)}")
    print(f"Total biosecurity goals found: {len(filtered_data)}")

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

if __name__ == "__main__":
    create_biosecurity_dataset()
