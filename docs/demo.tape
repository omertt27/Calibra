# Calibra Integrity demo recording.
#
# Requires vhs (https://github.com/charmbracelet/vhs): `brew install vhs`
# Run from the repo root:
#     vhs docs/demo.tape
# Produces docs/figures/integrity_demo.gif for the README / docs site.

Output docs/figures/integrity_demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1000
Set Height 560
Set Theme "Dracula"

Type "python docs/demo_fixture.py /tmp/calibra_demo.h5 && calibra integrity /tmp/calibra_demo.h5"
Enter
Sleep 4s

Type "# Once integrity passes, move to quality, coverage, and optimization:"
Enter
Sleep 1s
Type "calibra audit /tmp/calibra_demo.h5    # is it clean?"
Sleep 1s
Enter
Sleep 2s
Type "calibra review /tmp/calibra_demo.h5   # is it diverse?"
Sleep 1s
Enter
Sleep 2s
Type "calibra prune /tmp/calibra_demo.h5 --keep 0.3 --out coreset.json   # train faster"
Sleep 1s
Enter
Sleep 3s
