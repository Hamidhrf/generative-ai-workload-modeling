#!/bin/bash

# Import Grafana Dashboards Script
# Automatically imports all dashboard JSON files to Grafana

set +e

echo "=========================================="
echo "Importing Grafana Dashboards"
echo "=========================================="
echo ""

# Configuration
GRAFANA_URL="http://172.22.174.58:30030"
GRAFANA_USER="admin"
GRAFANA_PASS="admin"
DASHBOARD_DIR="../../dashboards"

# Change to script directory
cd "$(dirname "$0")"

# Check if Grafana is accessible
echo "[1/4] Checking Grafana connectivity..."
if ! curl -s -u "${GRAFANA_USER}:${GRAFANA_PASS}" "${GRAFANA_URL}/api/health" > /dev/null 2>&1; then
    echo "Error: Cannot connect to Grafana at ${GRAFANA_URL}"
    echo "Please ensure:"
    echo "  1. Grafana pod is running: kubectl get pods -n monitoring"
    echo "  2. NodePort service is accessible: kubectl get svc -n monitoring grafana"
    exit 1
fi
echo "Success: Grafana is accessible"
echo ""

# Check if dashboard directory exists
echo "[2/4] Checking dashboard files..."
if [ ! -d "${DASHBOARD_DIR}" ]; then
    echo "Error: Dashboard directory not found: ${DASHBOARD_DIR}"
    exit 1
fi

# Count dashboard files
DASHBOARD_COUNT=$(find "${DASHBOARD_DIR}" -name "*.json" -type f 2>/dev/null | wc -l)
if [ "${DASHBOARD_COUNT}" -eq 0 ]; then
    echo "Error: No dashboard JSON files found in ${DASHBOARD_DIR}"
    exit 1
fi
echo "Found ${DASHBOARD_COUNT} dashboard(s) to import"
echo ""

# Get datasource UID
echo "[3/4] Getting Prometheus datasource UID..."
DATASOURCE_UID=$(curl -s -u "${GRAFANA_USER}:${GRAFANA_PASS}" \
    "${GRAFANA_URL}/api/datasources" 2>/dev/null | \
    jq -r '.[] | select(.type=="prometheus") | .uid' 2>/dev/null | head -1)

if [ -z "${DATASOURCE_UID}" ]; then
    echo "Warning: Prometheus datasource not found. Using default."
    DATASOURCE_UID="prometheus"
fi
echo "Using datasource UID: ${DATASOURCE_UID}"
echo ""

# Import dashboards
echo "[4/4] Importing dashboards..."
SUCCESS_COUNT=0
FAIL_COUNT=0

for dashboard_file in "${DASHBOARD_DIR}"/*.json; do
    # Skip if not a file
    [ -f "${dashboard_file}" ] || continue
    
    dashboard_name=$(basename "${dashboard_file}")
    echo "Importing: ${dashboard_name}..."
    
    # Extract dashboard object from JSON file
    dashboard_content=$(cat "${dashboard_file}" | jq -c '.dashboard' 2>/dev/null)
    
    if [ -z "${dashboard_content}" ] || [ "${dashboard_content}" = "null" ]; then
        echo "  Error: Could not extract dashboard from ${dashboard_name}"
        ((FAIL_COUNT++))
        echo ""
        continue
    fi
    
    # Create import payload using simpler method
    cat > /tmp/dashboard_import.json <<EOF
{
  "dashboard": ${dashboard_content},
  "overwrite": true,
  "inputs": [
    {
      "name": "DS_PROMETHEUS",
      "type": "datasource",
      "pluginId": "prometheus",
      "value": "${DATASOURCE_UID}"
    }
  ]
}
EOF
    
    # Import dashboard
    response=$(curl -s -u "${GRAFANA_USER}:${GRAFANA_PASS}" \
        -X POST \
        -H "Content-Type: application/json" \
        "${GRAFANA_URL}/api/dashboards/import" \
        -d @/tmp/dashboard_import.json 2>&1)
    
    # Check if import was successful
    if echo "${response}" | grep -q '"imported":true'; then
        echo "  Success: ${dashboard_name} imported"
        ((SUCCESS_COUNT++))
    else
        echo "  Failed: ${dashboard_name}"
        echo "  Response: ${response}"
        ((FAIL_COUNT++))
    fi
    echo ""
done

# Cleanup temp file
rm -f /tmp/dashboard_import.json

echo "=========================================="
echo "Import Complete"
echo "=========================================="
echo "Successfully imported: ${SUCCESS_COUNT} dashboard(s)"
echo "Failed: ${FAIL_COUNT} dashboard(s)"
echo ""

if [ "${SUCCESS_COUNT}" -gt 0 ]; then
    echo "Access dashboards at:"
    echo "  ${GRAFANA_URL}/dashboards"
    echo ""
    echo "Imported dashboards:"
    curl -s -u "${GRAFANA_USER}:${GRAFANA_PASS}" \
        "${GRAFANA_URL}/api/search?type=dash-db" 2>/dev/null | \
        jq -r '.[] | select(.title | contains("Thesis")) | "  - " + .title' 2>/dev/null | sort
    echo ""
fi

if [ "${FAIL_COUNT}" -gt 0 ]; then
    echo "Warning: Some dashboards failed to import."
    echo "Check Grafana logs: kubectl logs -n monitoring -l app=grafana"
    exit 1
fi

echo "Dashboard import successful!"
echo "=========================================="