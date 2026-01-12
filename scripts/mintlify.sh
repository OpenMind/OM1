# chmod +x mintlify.sh
# invoke with ./mintlify.sh

#!/bin/bash
set -e

# Step 1: Copy docs to mintlify/
cp -r docs/* mintlify/

# Step 2: Remove all ".mdx" strings from inside the files
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS-compatible sed with LC_CTYPE fix
  find mintlify -type f -exec sh -c 'LC_CTYPE=C sed -i "" "s/\\.mdx//g" "$0"' {} \;
else
  # Linux-compatible
  find mintlify -type f -exec sed -i 's/\.mdx//g' {} +
fi
