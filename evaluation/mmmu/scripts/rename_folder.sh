#!/bin/bash

#!/bin/bash

cd /fs/nexus-projects/wilddiffusion/vlm/qwen_mcq || exit

# Rename all qwen2vl-3b* to qwen25_3b*
for f in qwen2vl-3b-*; do
    if [ -d "$f" ]; then
        newname="${f/qwen2vl-3b-/qwen25_3b_}"
        echo "Renaming $f -> $newname"
        mv "$f" "$newname"
    fi
done