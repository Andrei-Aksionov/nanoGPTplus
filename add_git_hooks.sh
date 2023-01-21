#!/bin/bash

git_hook_path=.git/hooks/pre-commit

echo "#!/bin/sh\n\nblack .\nflake8" > "$git_hook_path"

chmod +x "$git_hook_path"
