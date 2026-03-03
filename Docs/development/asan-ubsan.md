# AddressSanitizer and UndefinedBehaviorSanitizer

Run the NeuronDB test suite under sanitizers to catch memory and undefined behavior bugs.

## Build with ASan (AddressSanitizer)

```bash
# Build PostgreSQL and NeuronDB with ASan
export CFLAGS="-fsanitize=address -fno-omit-frame-pointer -g"
export LDFLAGS="-fsanitize=address"
# Then build Postgres and the extension as usual (see INSTALL.md)
```

Run the SQL test suite; any heap buffer overflow or use-after-free will abort with a report.

## Build with UBSan (UndefinedBehaviorSanitizer)

```bash
export CFLAGS="-fsanitize=undefined -fno-omit-frame-pointer -g"
export LDFLAGS="-fsanitize=undefined"
# Build and run tests
```

## CI

Add a job in CI that builds with ASan (or UBSan), runs the extension installcheck/sql tests, and fails on sanitizer errors. Ensure `continue-on-error` is false for this job so findings block the merge.
