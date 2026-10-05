# Security policy

Version 0.1.x receives fixes as issues are identified. Report vulnerabilities privately through GitHub **Security → Report a vulnerability** when available, or use the maintainer contact shown on the repository owner's profile. Avoid public exploit details until a fix is available. Include affected version, reproduction steps, impact, and any suggested mitigation.

PrefillLab is a trusted local research application. The default server listens only on 127.0.0.1, with no user authentication. Do not expose it to untrusted networks without authentication and network controls. Real-model HTTP execution is opt-in through `PREFILLLAB_ALLOW_REAL_MODELS=1`; CLI and Python real runs are explicit. Remote Hugging Face model code is disabled.

Keep API requests, model paths, imported results, and shared database access within the trusted workspace. Imported result JSON is schema-validated; reports escape strings and do not execute scripts. Dependencies are optional where possible and CI uses a frontend lockfile. Benchmark environment exports may contain system identifiers, paths, or Git metadata; review them before sharing.

Never put access tokens, model-provider credentials, private weights, or environment files into an issue or repository commit.
