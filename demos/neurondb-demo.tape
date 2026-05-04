Output docs/assets/neurondb-demo.gif

Set Shell "bash"
Set FontSize 22
Set Width 110
Set Height 32
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "curl -fsSL https://raw.githubusercontent.com/neurondb/neurondb/main/scripts/install-docker.sh | bash"
Sleep 500ms
Enter
Sleep 14s

Type "psql \"postgresql://neurondb:neurondb@localhost:5433/neurondb\""
Sleep 400ms
Enter
Sleep 2s

Type "CREATE EXTENSION IF NOT EXISTS neurondb;"
Sleep 300ms
Enter
Sleep 1s

Type "SELECT neurondb.version();"
Sleep 300ms
Enter
Sleep 1s

Type "SELECT ARRAY[1.0, 2.0, 3.0]::vector(3);"
Sleep 300ms
Enter
Sleep 1s

Type "\\q"
Sleep 200ms
Enter
Sleep 1s
