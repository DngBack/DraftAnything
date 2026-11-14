# VATr++ Vietnamese handwriting research

The source trees are vendored as ordinary files so a normal clone includes every
model, trainer, script and test. No submodule initialization is required.

- [Reproduce the Vietnamese work](VATr-pp/REPRODUCE.md)
- [Notable weights and exact download/import instructions](VATr-pp/WEIGHTS.md)
- [Reports, algorithm explanations and current training](VATr-pp/docs/README.md)
- [Upstream source commits and provenance](UPSTREAM.json)

`VATr-pp` is the modified native implementation used for Vietnamese training.
`VATr` preserves the earlier reference implementation and its license. Virtual
environments, training images, caches and multi-GB run outputs are not code;
they are excluded from Git and documented separately. The original nested Git
metadata was backed up under the parent repository's `.git/vendor-backups/261114`
on the authoring machine instead of being deleted.
