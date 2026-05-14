
import os

file_path = r"c:\Users\EDWIN\OneDrive\Desktop\Semiparametric HNB\model\sp_hnb.py"

with open(file_path, 'rb') as f:
    content = f.read()

# content is bytes
# Check for BOM
if content.startswith(b'\xef\xbb\xbf'):
    print("Found UTF-8 BOM, removing it.")
    content = content[3:]

# Decode
text = content.decode('utf-8')
lines = text.splitlines()

# Fix Line 1
# Ensure it is exactly "import numpy as np" with no leading stuff
if lines[0].strip() == "import numpy as np":
    lines[0] = "import numpy as np"
    print("Reset line 1.")

# Fix bad indentation in _ucv_bandwidth
# We look for "        for h in h_grid:"
# And the following block indented with 2 extra spaces (total 10) instead of 4 (total 12).
fixed_lines = []
in_loop = False
loop_start_sentinel = "        for h in h_grid:"

for i, line in enumerate(lines):
    if line.rstrip() == loop_start_sentinel:
        in_loop = True
        fixed_lines.append(line)
        print(f"Found loop at line {i+1}")
        continue
    
    if in_loop:
        # Check if line is indented with 10 spaces
        # But wait, python might have 2-space indent
        if line.startswith("          ") and not line.startswith("            "):
            # Replace 10 spaces with 12
            fixed_line = "            " + line[10:]
            fixed_lines.append(fixed_line)
            # print(f"Fixed indent at line {i+1}")
        elif line.strip() == "":
             fixed_lines.append("")
        elif line.startswith("        ") and len(line) > 8 and line[8] != ' ': # End of loop (back to 8 spaces)
             in_loop = False
             fixed_lines.append(line)
             print(f"End of loop at line {i+1}")
        else:
            # Maybe inside loop but different indent?
            # If it's a deeper block inside loop
            if line.startswith("            "): # 12 spaces in original
                 # If original was 12 spaces (e.g. if inside the loop), it should be 16 now?
                 # Wait, strict 2 space indent meant 10 -> 12. 
                 # 12 -> 16? 
                 # Let's see if there are deeper blocks.
                 # "if ucv_score < best_score:" is at same level.
                 # "  best_score = ..." is at 12 spaces (original). Should be 16.
                 fixed_line = "    " + line
                 fixed_lines.append(fixed_line)
            elif line.startswith("          "): # already handled by first if
                 fixed_lines.append("            " + line[10:])
            else:
                 # Probably end of loop or empty line
                 if len(line.strip()) > 0 and not line.startswith("          "):
                     in_loop = False
                 fixed_lines.append(line)
    else:
        fixed_lines.append(line)

# Join and write back
new_content = "\n".join(fixed_lines)
with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
    f.write(new_content)

print("Done.")
