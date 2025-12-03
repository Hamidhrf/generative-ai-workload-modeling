# Troubleshooting Guide

## Common Issues

### Cluster won't start after reboot
\`\`\`bash
sudo systemctl status crio
sudo systemctl status kubelet
\`\`\`

### GPU not detected
\`\`\`bash
cd scripts/gpu-setup
sudo ./gpu-health-check.sh
\`\`\`
