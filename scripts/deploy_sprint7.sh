#!/bin/bash
# Sprint 7 Deployment Script
# Raspberry Pi 5 Federated Environmental Monitoring Network

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
INVENTORY_FILE="$SCRIPT_DIR/inventory.yml"
PLAYBOOK_FILE="$SCRIPT_DIR/provision_pi5.yml"
LOG_FILE="$PROJECT_ROOT/logs/deployment_$(date +%Y%m%d_%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

log_info() {
    log "INFO" "${BLUE}$@${NC}"
}

log_success() {
    log "SUCCESS" "${GREEN}$@${NC}"
}

log_warning() {
    log "WARNING" "${YELLOW}$@${NC}"
}

log_error() {
    log "ERROR" "${RED}$@${NC}"
}

# Function to check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check if Ansible is installed
    if ! command -v ansible-playbook &> /dev/null; then
        log_error "Ansible is not installed. Please install Ansible first."
        echo "Install with: pip install ansible"
        exit 1
    fi
    
    # Check if inventory file exists
    if [ ! -f "$INVENTORY_FILE" ]; then
        log_error "Inventory file not found: $INVENTORY_FILE"
        exit 1
    fi
    
    # Check if playbook exists
    if [ ! -f "$PLAYBOOK_FILE" ]; then
        log_error "Playbook file not found: $PLAYBOOK_FILE"
        exit 1
    fi
    
    # Check SSH connectivity
    log_info "Testing SSH connectivity to Pi nodes..."
    if ansible pi_nodes -i "$INVENTORY_FILE" -m ping --one-line > /dev/null 2>&1; then
        log_success "SSH connectivity to Pi nodes verified"
    else
        log_warning "SSH connectivity test failed. Nodes may not be accessible."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    log_success "Prerequisites check completed"
}

# Function to create necessary directories
setup_directories() {
    log_info "Setting up deployment directories..."
    
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/certificates"
    mkdir -p "$PROJECT_ROOT/data"
    
    log_success "Directories created"
}

# Function to generate SSL certificates
generate_certificates() {
    log_info "Generating SSL certificates for secure communication..."
    
    local cert_dir="$PROJECT_ROOT/certificates"
    
    # Generate CA private key
    if [ ! -f "$cert_dir/ca-key.pem" ]; then
        openssl genrsa -out "$cert_dir/ca-key.pem" 4096
        log_info "Generated CA private key"
    fi
    
    # Generate CA certificate
    if [ ! -f "$cert_dir/ca-cert.pem" ]; then
        openssl req -new -x509 -days 365 -key "$cert_dir/ca-key.pem" \
            -out "$cert_dir/ca-cert.pem" \
            -subj "/C=US/ST=CA/L=San Francisco/O=Raspberry Pi 5 Federated/CN=Federated Learning CA"
        log_info "Generated CA certificate"
    fi
    
    # Generate server certificate
    if [ ! -f "$cert_dir/server-cert.pem" ]; then
        openssl genrsa -out "$cert_dir/server-key.pem" 4096
        openssl req -new -key "$cert_dir/server-key.pem" \
            -out "$cert_dir/server-csr.pem" \
            -subj "/C=US/ST=CA/L=San Francisco/O=Raspberry Pi 5 Federated/CN=federated-server"
        openssl x509 -req -days 365 -in "$cert_dir/server-csr.pem" \
            -CA "$cert_dir/ca-cert.pem" -CAkey "$cert_dir/ca-key.pem" \
            -out "$cert_dir/server-cert.pem" -CAcreateserial
        rm "$cert_dir/server-csr.pem"
        log_info "Generated server certificate"
    fi
    
    # Generate client certificates for each Pi node
    for node in pi-node-01 pi-node-02 pi-node-03; do
        if [ ! -f "$cert_dir/client-${node}.crt" ]; then
            openssl genrsa -out "$cert_dir/client-${node}.key" 4096
            openssl req -new -key "$cert_dir/client-${node}.key" \
                -out "$cert_dir/client-${node}.csr" \
                -subj "/C=US/ST=CA/L=San Francisco/O=Raspberry Pi 5 Federated/CN=${node}"
            openssl x509 -req -days 365 -in "$cert_dir/client-${node}.csr" \
                -CA "$cert_dir/ca-cert.pem" -CAkey "$cert_dir/ca-key.pem" \
                -out "$cert_dir/client-${node}.crt" -CAcreateserial
            rm "$cert_dir/client-${node}.csr"
            log_info "Generated client certificate for $node"
        fi
    done
    
    log_success "SSL certificates generated"
}

# Function to deploy to Pi nodes
deploy_pi_nodes() {
    log_info "Starting deployment to Raspberry Pi 5 nodes..."
    
    # Run Ansible playbook
    if ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_FILE" \
        --extra-vars "project_root=$PROJECT_ROOT" \
        -v | tee -a "$LOG_FILE"; then
        log_success "Ansible playbook execution completed"
    else
        log_error "Ansible playbook execution failed"
        return 1
    fi
}

# Function to verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Check if services are running
    log_info "Checking service status on all nodes..."
    
    for node in pi-node-01 pi-node-02 pi-node-03; do
        log_info "Checking $node..."
        
        if ansible "$node" -i "$INVENTORY_FILE" \
            -m shell -a "systemctl is-active federated-client-$node" \
            --one-line | grep -q "SUCCESS"; then
            log_success "$node: Service is running"
        else
            log_warning "$node: Service may not be running properly"
        fi
        
        # Check sensor connectivity
        if ansible "$node" -i "$INVENTORY_FILE" \
            -m shell -a "cd /home/pi/raspberry-pi5-federated && ./venv/bin/python scripts/test_sensors.py" \
            --one-line > /dev/null 2>&1; then
            log_success "$node: Sensor test passed"
        else
            log_warning "$node: Sensor test failed"
        fi
    done
}

# Function to start 24-hour validation test
start_validation_test() {
    log_info "Starting 24-hour validation test..."
    
    # Create validation script
    cat > "$PROJECT_ROOT/scripts/validation_test.sh" << 'EOF'
#!/bin/bash
# 24-hour validation test for Sprint 7

VALIDATION_LOG="$1"
DURATION_HOURS=24
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION_HOURS * 3600))

echo "$(date): Starting 24-hour validation test" >> "$VALIDATION_LOG"

while [ $(date +%s) -lt $END_TIME ]; do
    # Check system health every 5 minutes
    for node in pi-node-01 pi-node-02 pi-node-03; do
        # Check service status
        if ansible "$node" -i inventory.yml -m shell -a "systemctl is-active federated-client-$node" --one-line | grep -q "SUCCESS"; then
            echo "$(date): $node service OK" >> "$VALIDATION_LOG"
        else
            echo "$(date): $node service FAILED" >> "$VALIDATION_LOG"
        fi
        
        # Check system resources
        ansible "$node" -i inventory.yml -m shell -a "free -m | grep '^Mem:' | awk '{print \$3/\$2*100}'" --one-line >> "$VALIDATION_LOG" 2>/dev/null
    done
    
    sleep 300  # 5 minutes
done

echo "$(date): 24-hour validation test completed" >> "$VALIDATION_LOG"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/validation_test.sh"
    
    # Start validation test in background
    nohup "$PROJECT_ROOT/scripts/validation_test.sh" "$PROJECT_ROOT/logs/validation_test.log" > /dev/null 2>&1 &
    local validation_pid=$!
    
    echo "$validation_pid" > "$PROJECT_ROOT/logs/validation_test.pid"
    
    log_success "24-hour validation test started (PID: $validation_pid)"
    log_info "Monitor progress with: tail -f $PROJECT_ROOT/logs/validation_test.log"
}

# Function to display deployment summary
show_summary() {
    log_info "Deployment Summary"
    echo "===========================================" | tee -a "$LOG_FILE"
    echo "Sprint 7: Pilot Deployment & Validation" | tee -a "$LOG_FILE"
    echo "Deployment completed at: $(date)" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Deployed Nodes:" | tee -a "$LOG_FILE"
    echo "  - pi-node-01 (192.168.1.101) - Environmental Monitor" | tee -a "$LOG_FILE"
    echo "  - pi-node-02 (192.168.1.102) - Vibration Monitor" | tee -a "$LOG_FILE"
    echo "  - pi-node-03 (192.168.1.103) - Environmental Monitor" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Next Steps:" | tee -a "$LOG_FILE"
    echo "  1. Monitor 24-hour validation test: tail -f logs/validation_test.log" | tee -a "$LOG_FILE"
    echo "  2. Check node status: ansible pi_nodes -i scripts/inventory.yml -m shell -a 'systemctl status federated-client-*'" | tee -a "$LOG_FILE"
    echo "  3. View logs: ssh pi@<node-ip> 'tail -f raspberry-pi5-federated/logs/*.log'" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "===========================================" | tee -a "$LOG_FILE"
}

# Main deployment function
main() {
    log_info "Starting Sprint 7 deployment process"
    log_info "Raspberry Pi 5 Federated Environmental Monitoring Network"
    
    # Check command line arguments
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        echo "Usage: $0 [--skip-checks] [--skip-validation]"
        echo ""
        echo "Options:"
        echo "  --skip-checks      Skip prerequisite checks"
        echo "  --skip-validation  Skip 24-hour validation test"
        echo "  --help, -h         Show this help message"
        exit 0
    fi
    
    # Parse arguments
    SKIP_CHECKS=false
    SKIP_VALIDATION=false
    
    for arg in "$@"; do
        case $arg in
            --skip-checks)
                SKIP_CHECKS=true
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                ;;
        esac
    done
    
    # Execute deployment steps
    if [ "$SKIP_CHECKS" = false ]; then
        check_prerequisites
    fi
    
    setup_directories
    generate_certificates
    deploy_pi_nodes
    verify_deployment
    
    if [ "$SKIP_VALIDATION" = false ]; then
        start_validation_test
    fi
    
    show_summary
    
    log_success "Sprint 7 deployment completed successfully!"
}

# Run main function with all arguments
main "$@"
