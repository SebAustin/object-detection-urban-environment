#!/usr/bin/env bash
#
# Execute one project notebook on the SageMaker notebook instance and save its outputs IN PLACE,
# so the submitted .ipynb shows every executed cell (the reviewer requires this).
#
# Run it from a Jupyter terminal, inside the repository, under nohup so that a closed browser tab
# or an expired Cloud Lab session doesn't kill the run:
#
#   nohup tools/run_notebook.sh 1_model_training/1_train_model_efficientdet_d1.ipynb > run-effdet.log 2>&1 &
#   tail -f run-effdet.log
#
# Executing in a fresh kernel means the "restart the kernel after pip install" step is not needed:
# the install cell runs before anything imports sagemaker.
#
# Safety: if the deploy notebook fails after creating an endpoint, every endpoint (plus its config
# and model) whose name contains "tf2-object-detection" is deleted, so it can't keep billing.

set -euo pipefail

notebook=${1:?"Usage: tools/run_notebook.sh <path/to/notebook.ipynb>"}
[ -f "${notebook}" ] || { echo "ERROR: ${notebook} not found" >&2; exit 1; }

KERNEL=${KERNEL:-conda_python3}
ENDPOINT_MARKER="tf2-object-detection"

cleanup_endpoints() {
    echo "== Deleting leftover endpoints containing '${ENDPOINT_MARKER}' =="
    for name in $(aws sagemaker list-endpoints --name-contains "${ENDPOINT_MARKER}" \
                    --query 'Endpoints[].EndpointName' --output text); do
        echo "deleting ${name}"
        aws sagemaker delete-endpoint --endpoint-name "${name}" || true
        aws sagemaker delete-endpoint-config --endpoint-config-name "${name}" || true
        aws sagemaker delete-model --model-name "${name}" || true
    done
}

echo "== $(date -u +%FT%TZ) executing ${notebook} with kernel ${KERNEL} =="
status=0
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name="${KERNEL}" \
    --ExecutePreprocessor.timeout=-1 \
    "${notebook}" || status=$?

if [ "${status}" -ne 0 ]; then
    echo "== FAILED (exit ${status}). Outputs up to the failing cell are saved in ${notebook}. =="
    case "${notebook}" in
        *2_deploy_model.ipynb) cleanup_endpoints ;;
    esac
    exit "${status}"
fi
echo "== $(date -u +%FT%TZ) done: ${notebook} now contains its outputs =="
