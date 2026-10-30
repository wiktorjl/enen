#!/bin/sh
set -eu

# Each command uses private fixtures, including a tiny configuration so that
# tuning all sixteen recipes is fast and never touches the real model or data.
project_dir=$(pwd)
binary_dir=${ENEN_TEST_BIN_DIR:-"$project_dir/build"}
fixture_dir=$(mktemp -d "${TMPDIR:-/tmp}/enen-cli-test.XXXXXX")
trap 'rm -rf "$fixture_dir"' EXIT HUP INT TERM

fail() {
    echo "FAIL: $*" >&2
    if [ -f "$fixture_dir/command.log" ]; then
        cat "$fixture_dir/command.log" >&2
    fi
    exit 1
}

expect_failure() {
    if "$@" > "$fixture_dir/command.log" 2>&1; then
        fail "command should fail: $*"
    fi
}

head -n 1 datasets/optdigits.tra > "$fixture_dir/original.tra"
cp "$fixture_dir/original.tra" "$fixture_dir/raw.tra"
converter="$binary_dir/convert_optdigits"

expect_failure "$converter" "$fixture_dir/raw.tra" "$fixture_dir/./raw.tra"
cmp "$fixture_dir/original.tra" "$fixture_dir/raw.tra"
ln -s raw.tra "$fixture_dir/alias.tra"
expect_failure "$converter" "$fixture_dir/raw.tra" "$fixture_dir/alias.tra"
cmp "$fixture_dir/original.tra" "$fixture_dir/raw.tra"
ln "$fixture_dir/raw.tra" "$fixture_dir/hardlink.tra"
expect_failure "$converter" "$fixture_dir/raw.tra" "$fixture_dir/hardlink.tra"
cmp "$fixture_dir/original.tra" "$fixture_dir/hardlink.tra"

echo 'earlier valid output' > "$fixture_dir/previous.csv"
cp "$fixture_dir/previous.csv" "$fixture_dir/output.csv"
echo 'malformed input' > "$fixture_dir/invalid.tra"
expect_failure "$converter" "$fixture_dir/invalid.tra" "$fixture_dir/output.csv"
cmp "$fixture_dir/previous.csv" "$fixture_dir/output.csv"
mkdir "$fixture_dir/destination"
expect_failure "$converter" "$fixture_dir/raw.tra" "$fixture_dir/destination"
[ -d "$fixture_dir/destination" ] || fail 'failed rename removed the destination'
for leftover in "$fixture_dir"/*.tmp.*; do
    [ ! -e "$leftover" ] || fail 'failed conversion left a temporary file'
done

cd "$fixture_dir"
mkdir conf models
cat > conf/digits.conf <<'EOF'
input_size=2
hidden_layers=3
output_size=2
learning_rate=0.05
epochs=1
train_dataset=train.csv
test_dataset=test.csv
EOF
cat > train.csv <<'EOF'
0,0,0
1,1,1
0,0.5,0
1,0.5,1
0.1,0.2,0
0.9,0.8,1
EOF
cat > test.csv <<'EOF'
0.3,0.1,0
0.7,0.9,1
EOF
cat > validation.csv <<'EOF'
0.2,0.3,0
0.8,0.7,1
EOF

"$binary_dir/digits" > command.log 2>&1
expect_failure "$binary_dir/gym" --load models/digits.model
grep -q -- '--load requires --validation' command.log || fail 'missing validation guard'
cp train.csv copied-training.csv
expect_failure "$binary_dir/gym" --load models/digits.model --validation copied-training.csv
grep -q 'overlap' command.log || fail 'copied training rows were not rejected'
cp test.csv copied-test.csv
expect_failure "$binary_dir/gym" --load models/digits.model --validation copied-test.csv
grep -q 'overlap' command.log || fail 'copied test rows were not rejected'

"$binary_dir/gym" --load models/digits.model --validation validation.csv > command.log 2>&1
grep -q 'Fitting 6 samples; validating on 2 samples.' command.log || fail 'independent validation failed'
"$binary_dir/gym" > command.log 2>&1
grep -q 'Fitting 4 samples; validating on 2 samples.' command.log || fail 'default holdout failed'

echo 'All CLI safety and validation tests passed.'
