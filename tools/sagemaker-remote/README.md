# SageMaker + Cursor on your PC

This folder documents how **your machine** (Cursor, SSH clients, Git) fits together with **AWS SageMaker**. Nothing here runs automatically in AWS; it is the “operator’s handbook” for this repo.

## What you are running today

- **SageMaker Notebook Instance** `digitalaglab-tomato-alphaearth` (`ml.g4dn.xlarge`, JupyterLab in the browser).
- That is **not** the same product as **SageMaker Studio** *spaces*.

## Important limitation (SSH / Remote IDE)

AWS’s supported **“Remote IDE + SSH”** flow for **Cursor** uses **SageMaker Studio spaces** and **Session Manager** (see [Connect your Remote IDE to SageMaker spaces with remote access](https://docs.aws.amazon.com/sagemaker/latest/dg/remote-access.html)).

**Classic Notebook Instances** do not expose the same integration. There is no simple “add EC2 key pair and paste into `~/.ssh/config`” path documented for them like there is for **Studio + SSM**.

So:

| You want | Realistic approach |
|----------|-------------------|
| **GPU Jupyter now** | Keep using **browser JupyterLab** on the notebook instance (what you have). |
| **Cursor as editor, code on GPU** | Prefer **Studio space + remote access** (AWS Toolkit + Remote-SSH + SSM plugin), or a **separate EC2 GPU** with an SSH key and **Cursor Remote-SSH**. |
| **Hacks on Notebook Instance** | Community tools (e.g. [sagemaker-ssh-helper](https://github.com/aws-samples/sagemaker-ssh-helper)) — extra IAM/lifecycle; not covered by our infra template. |

## Organized layout on your PC (recommended)

Use **one Git clone** of this repo on your laptop and treat it as source of truth:

```text
C:\mnarimani\1-UCDavis\9-Github\Google_AlphaEarth_Tomato_Farms\
  infra\                   # CloudFormation for notebook + bucket
  tools\sagemaker-remote\  # This guide
  notebooks\
  src\
  ...
```

On SageMaker (Jupyter terminal):

1. Clone the **same** repo (or `git pull` after you push from home).
2. Keep **heavy GeoTIFFs** in **S3** (`tomato-alphaearth-<ACCOUNT_ID>-data` from the stack, or your own bucket), not only on the notebook disk.

Workflow: push from Cursor when you finish an edit → pull on SageMaker before you train.

## Local prerequisites for **Studio** remote access (when you adopt it)

Install **before** using AWS’s Cursor connection to a **Studio space** (not required for browser-only notebook):

1. [Session Manager plugin for AWS CLI](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager-working-with-install-plugin.html) (Windows: AWS-provided `.exe` installer).
2. **OpenSSH Client** (Windows optional feature).
3. In **Cursor**:
   - **Remote - SSH** (`ms-vscode-remote.remote-ssh`)
   - **AWS Toolkit** (Amazon’s extension; version per [AWS remote access docs](https://docs.aws.amazon.com/sagemaker/latest/dg/remote-access.html)).
4. **Cursor** ≥ **2.6.18** (per AWS).

Then follow: [Set up Remote IDE](https://docs.aws.amazon.com/sagemaker/latest/dg/remote-access-local-ide-setup.html) and your admin’s Studio setup.

## Jupyter from Cursor against SageMaker (limited)

Some teams point the **Jupyter** extension at a remote URL. **Notebook Instance** URLs are tied to **AWS/IAM sign-in**, not a static token, so this is often **less reliable** than browser Jupyter or Studio remote IDE. Prefer browser Jupyter unless you have a documented URL pattern that works with your SSO.

## See also

- Notebook + bucket IaC: [`infra/README.md`](../../infra/README.md)
- Optional SSH config template (for generic EC2 or future Studio-generated hosts): [`ssh-config.SNIPPET.txt`](./ssh-config.SNIPPET.txt)
