#!/bin/bash
# Add the scopedsync category to a cuHadron checkout: ./add_category.sh /path/to/cuHadron
set -e
D="$1"; HERE="$(cd "$(dirname "$0")" && pwd)"
cp -r "$HERE/scopedsync" "$D/"
grep -q '^scopedsync ' "$D/test_requirements.txt" || echo "scopedsync 80 1" >> "$D/test_requirements.txt"
sed -i 's/^TEST_CATEGORIES=(\(.*\) interwave)/TEST_CATEGORIES=(\1 interwave scopedsync)/' "$D/run_tests.sh"
sed -i 's/^CATEGORY_PATTERN="\(.*\)|interwave"/CATEGORY_PATTERN="\1|interwave|scopedsync"/' "$D/run_tests.sh"
grep -n "scopedsync" "$D/run_tests.sh" "$D/test_requirements.txt"
