#!/usr/bin/env bash
#
# Run the whole experiment headlessly on a SageMaker notebook instance, in order:
#   Run 1 (EfficientDet D1) -> pause for review -> Run 2 -> Run 3 -> summary -> deploy + video.
# Each notebook is executed in place by tools/run_notebook.sh, so it keeps every output.
#
# Started by the notebook instance's lifecycle configuration (as ec2-user, under nohup):
#   tools/sagemaker_pipeline.sh s3://<bucket>/project-runs/<run-id> <notebook-instance-name> [all|deploy-only]
#
# Progress and artifacts go to that S3 prefix:
#   status.txt        current stage (or DONE / FAILED: ...)
#   pipeline.log      full log
#   outputs/...       executed notebooks, results/, output.avi (uploaded after each stage)
# Gate after Run 1: the pipeline waits until  <prefix>/control/continue  exists
# (or stops if  <prefix>/control/abort  exists), so Run 1 can be checked before spending more.
# On exit (success or failure) it stops the notebook instance so nothing keeps billing.

set -uo pipefail

S3_PREFIX=${1:?"Usage: sagemaker_pipeline.sh s3://bucket/prefix <notebook-instance-name>"}
INSTANCE_NAME=${2:?"Usage: sagemaker_pipeline.sh s3://bucket/prefix <notebook-instance-name>"}
# "all" (default) runs every stage; "deploy-only" re-runs just the deploy notebook, reusing the
# executed training/summary notebooks and results/ already on this instance.
MODE=${3:-all}
case "${MODE}" in all|deploy-only) ;; *) echo "Unknown mode '${MODE}' (use all or deploy-only)" >&2; exit 2 ;; esac
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GATE_POLL_SECONDS=60
GATE_TIMEOUT_SECONDS=$((3 * 3600))
KERNEL_NAME=project_py3

# Pin the region for every aws/boto3 call, including the self-stop in finish(). A notebook instance
# started by a lifecycle script does not always have one configured, and without it both the
# SageMaker SDK and the self-stop fail. Fallback order: env, aws config, instance metadata, us-east-1.
imds_region() {
    local token
    token=$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token \
        -H "X-aws-ec2-metadata-token-ttl-seconds: 60") || return 0
    curl -s -m 2 -H "X-aws-ec2-metadata-token: ${token}" \
        http://169.254.169.254/latest/meta-data/placement/region || true
}
REGION=${AWS_DEFAULT_REGION:-${AWS_REGION:-}}
[ -n "${REGION}" ] || REGION=$(aws configure get region 2>/dev/null || true)
[ -n "${REGION}" ] || REGION=$(imds_region)
[ -n "${REGION}" ] || REGION=us-east-1
export AWS_DEFAULT_REGION=${REGION} AWS_REGION=${REGION}

# The conda env behind the "conda_python3" kernel; its location differs between platform versions.
PY_ENV=""
for candidate in /home/ec2-user/anaconda3/envs/python3 /home/ec2-user/miniconda3/envs/python3 /opt/conda/envs/python3; do
    if [ -x "${candidate}/bin/python" ]; then PY_ENV=${candidate}; break; fi
done
if [ -z "${PY_ENV}" ]; then
    found=$(find /home/ec2-user /opt -maxdepth 5 -path '*envs/python3/bin/python' 2>/dev/null | head -1)
    [ -n "${found}" ] && PY_ENV=$(dirname "$(dirname "${found}")")
fi

cd "${REPO_DIR}" || exit 1
LOG="${REPO_DIR}/pipeline.log"
exec > >(tee -a "${LOG}") 2>&1

status() {
    echo "== $(date -u +%FT%TZ) STATUS: $1"
    echo "$1" | aws s3 cp - "${S3_PREFIX}/status.txt" --quiet
}

upload() {
    aws s3 cp "${LOG}" "${S3_PREFIX}/pipeline.log" --quiet
    for nb in 1_model_training/*.ipynb 2_run_inference/2_deploy_model.ipynb; do
        aws s3 cp "${nb}" "${S3_PREFIX}/outputs/${nb}" --quiet
    done
    [ -d 1_model_training/results ] && aws s3 sync 1_model_training/results "${S3_PREFIX}/outputs/1_model_training/results" --quiet
    [ -f 2_run_inference/output.avi ] && aws s3 cp 2_run_inference/output.avi "${S3_PREFIX}/outputs/2_run_inference/output.avi" --quiet
    return 0
}

finish() {
    upload
    echo "== stopping notebook instance ${INSTANCE_NAME}"
    aws sagemaker stop-notebook-instance --notebook-instance-name "${INSTANCE_NAME}" || true
}
trap finish EXIT

run() {
    local nb=$1
    status "RUNNING ${nb}"
    if ! KERNEL="${KERNEL_NAME}" PATH="${PY_ENV}/bin:${PATH}" tools/run_notebook.sh "${nb}"; then
        status "FAILED: ${nb}"
        exit 1
    fi
    upload
}

wait_for_gate() {
    status "WAITING_FOR_REVIEW after Run 1 (create ${S3_PREFIX}/control/continue to go on)"
    local waited=0
    while [ "${waited}" -lt "${GATE_TIMEOUT_SECONDS}" ]; do
        if aws s3 ls "${S3_PREFIX}/control/abort" > /dev/null 2>&1; then
            status "ABORTED at review gate"
            exit 1
        fi
        if aws s3 ls "${S3_PREFIX}/control/continue" > /dev/null 2>&1; then
            return 0
        fi
        sleep "${GATE_POLL_SECONDS}"
        waited=$((waited + GATE_POLL_SECONDS))
    done
    status "ABORTED: review gate timed out"
    exit 1
}

status "SETUP"
if [ -z "${PY_ENV}" ]; then
    status "FAILED: no conda python3 environment found"
    exit 1
fi
echo "Using Python env ${PY_ENV}, region ${AWS_DEFAULT_REGION}"
"${PY_ENV}/bin/python" --version
docker --version
"${PY_ENV}/bin/pip" install -q nbconvert nbclient ipykernel
"${PY_ENV}/bin/python" -m ipykernel install --user --name "${KERNEL_NAME}" --display-name "conda_python3 (pipeline)"

if [ "${MODE}" = "all" ]; then
    run 1_model_training/1_train_model_efficientdet_d1.ipynb
    wait_for_gate
    run 1_model_training/1_train_model_ssd_mobilenet_v2_fpnlite_640.ipynb
    run 1_model_training/1_train_model_ssd_resnet50_v1_fpn_640.ipynb
    run 1_model_training/3_experiment_summary.ipynb
fi
run 2_run_inference/2_deploy_model.ipynb
status "DONE"
