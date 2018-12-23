python compile_all.py
mkdir ../pyc_repo
find ./ -not -name '*.py' -not -path "./output/*" -not -path "./.git/*" -not -path "./.idea/*" | xargs cp --parents -t ../pyc_repo
find ./ -path '*/symbols/*.py' | xargs cp --parents -t ../pyc_repo
