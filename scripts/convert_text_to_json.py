import re
import json

# Change the argument to 'file_path' so the function is reusable
def markdown_to_structured_json(file_path="test.md"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return None

    # Split by H1 (#) or H2 (##) headers
    sections = re.split(r'\n(?=#{1,2} )', content)
    
    structured_data = []
    current_page = "Nvidia Manual"
    
    for section in sections:
        if not section.strip():
            continue
            
        lines = section.strip().split('\n')
        header_line = lines[0]
        title = header_line.replace('#', '').strip()
        
        body_content = '\n'.join(lines[1:]).strip()
        
        # Categorization logic
        category = "Technical"
        lower_title = title.lower()
        if any(word in lower_title for word in ["profile", "financial", "leadership", "revenue", "stock"]):
            category = "Corporate"
        elif "troubleshooting" in lower_title or "faults" in lower_title:
            category = "Support"

        entry = {
            "page_title": current_page,
            "section_title": title,
            "chunk_text": body_content,
            "metadata": {
                "category": category,
                "char_count": len(body_content),
                # Fixed the backtick check here
                "has_code_block": "```" in body_content
            }
        }
        structured_data.append(entry)
    
    return structured_data

def enrich_data(simple_json_list):
    enriched_data = []
    
    for i, chunk in enumerate(simple_json_list):
        # 1. Logic for Breadcrumbs
        # We can build this from the category and titles
        category = chunk["metadata"]["category"]
        breadcrumbs = ["Nvidia Manual", category, chunk["section_title"]]
        
        # 2. Logic for Context (The "Pro" move)
        # Grab text from the chunk before and after if they exist
        prev_text = simple_json_list[i-1]["chunk_text"][:500] if i > 0 else ""
        next_text = simple_json_list[i+1]["chunk_text"][:500] if i < len(simple_json_list)-1 else ""
        
        # 3. Create the "Pro" Payload
        enriched_payload = {
            "page_title": chunk["page_title"],
            "section_title": chunk["section_title"],
            "page_url": "https://nvidia.com/manual", # Base URL
            "section_url": f"https://nvidia.com/manual#{chunk['section_title'].replace(' ', '-').lower()}",
            "breadcrumbs": breadcrumbs,
            "chunk_text": chunk["chunk_text"],
            "prev_section_text": prev_text,
            "next_section_text": next_text,
            "tags": [category.lower(), "hardware" if category == "Technical" else "business"]
        }
        enriched_data.append(enriched_payload)
        
    return enriched_data

# --- EXECUTION ---
# Ensure your file is named 'nvidia_manual.md' or change this string
input_filename = 'data/nvidia_manual.md' 
json_data = markdown_to_structured_json(input_filename)

if json_data:
    # Enrich the data
    pro_data = enrich_data(json_data)
    
    output_file = 'data/nvidia_structured_docs.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pro_data, f, indent=4)

    print(f"✅ Success! Created {len(pro_data)} enriched chunks.")
    print(f"📂 File saved as: {output_file}")
else:
    print(f"❌ Error: '{input_filename}' not found. Please create the file first!")
