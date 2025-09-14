# SLURM Job Scheduler Agent

The SLURM Job Scheduler Agent is an intelligent automation system for managing HPC job submissions and monitoring in SLURM-based cluster environments. It integrates seamlessly with the DFT Agent workflow to provide end-to-end automation of computational materials science calculations.

## 🎯 Overview

The SLURM Scheduler Agent provides:
- **Automated Job Script Generation**: Creates optimized SLURM scripts for Quantum ESPRESSO calculations
- **Intelligent Job Submission**: Handles job submission with proper error checking and validation
- **Real-time Job Monitoring**: Tracks job status, runtime, and completion
- **Workflow Integration**: Coordinates with DFT calculations and other agents
- **HPC Resource Management**: Optimizes resource allocation and queue management

## 🏗️ Architecture

### Core Components

1. **SLURM Tools** (`backend/agents/dft_tools/slurm_tools.py`)
   - Job script generation and customization
   - Job submission and management
   - Status monitoring and output retrieval
   - Queue management and job cancellation

2. **Scheduler Agent** (`backend/agents/library/slurm_scheduler/`)
   - Intelligent job scheduling logic
   - Workflow coordination
   - Error handling and recovery
   - Integration with other agents

3. **Tool Registry Integration**
   - Seamless integration with existing DFT tools
   - Consistent API across all agents
   - Centralized tool management

## 🚀 Quick Start

### Basic Usage

```python
# Access the SLURM scheduler agent
from backend.agents.agent_manager import get_agent

scheduler = get_agent("slurm_scheduler")

# Generate and submit a job
result = scheduler.invoke({
    "messages": [HumanMessage(content="Generate and submit a SLURM job for scf.in")],
    "thread_id": "my_calculation"
})
```

### Via API

```bash
# Submit a job through the API
curl -X POST http://localhost:8083/agent/slurm_scheduler/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Generate and submit a SLURM job for my QE calculation",
    "thread_id": "calculation_001"
  }'
```

## 🛠️ Available Tools

### 1. Script Generation

**`generate_slurm_script`**
- Creates optimized SLURM job scripts
- Customizable resource allocation
- Automatic module loading
- Error handling and validation

```python
# Example usage
script_path = generate_slurm_script.invoke({
    "job_name": "pt_slab_scf",
    "input_file": "scf.in",
    "output_file": "scf.out",
    "partition": "GPU-S",
    "ntasks_per_node": 16,
    "cpus_per_task": 1,
    "time_limit": "72:00:00",
    "memory": "32G",
    "thread_id": "my_workspace"
})
```

### 2. Job Submission

**`submit_slurm_job`**
- Submits jobs to SLURM queue
- Validates script before submission
- Tracks job information
- Handles submission errors

```python
# Submit a job
result = submit_slurm_job.invoke({
    "script_path": "/path/to/job.sh",
    "thread_id": "my_workspace"
})
```

### 3. Job Monitoring

**`check_slurm_job_status`**
- Real-time job status checking
- Runtime and resource information
- Queue position tracking
- Historical job data

```python
# Check job status
status = check_slurm_job_status.invoke({
    "job_id": "12345",
    "thread_id": "my_workspace"
})
```

### 4. Job Management

**`cancel_slurm_job`**
- Cancel running or pending jobs
- Update job tracking information
- Clean up resources

**`list_slurm_jobs`**
- List all jobs in queue
- Filter by user, partition, or status
- Comprehensive job information

**`get_slurm_job_output`**
- Retrieve job output files
- Parse calculation results
- Handle large output files

**`monitor_slurm_jobs`**
- Monitor multiple jobs simultaneously
- Automated status updates
- Workspace-specific monitoring

## 📋 SLURM Script Template

The agent uses a customizable template for generating SLURM scripts:

```bash
#!/bin/bash
#SBATCH --partition=GPU-S
#SBATCH --job-name=<job_name>
#SBATCH --output=<output_file>
#SBATCH --error=<error_file>
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --requeue
#SBATCH --mem=32G

echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "Working Directory=$SLURM_SUBMIT_DIR"
echo "Job started at: $(date)"

# Load required modules
module load compiler-rt/latest
module load mkl/latest
module load mpi/latest

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
ulimit -s unlimited

# Change to job directory
cd $SLURM_SUBMIT_DIR

# Run the calculation
mpiexec -np $SLURM_NTASKS_PER_NODE /cm/shared/apps/quantumespresso/qe-7.2/bin/pw.x -i <input_file>

echo "Job completed at: $(date)"
```

## 🔧 Configuration

### Environment Variables

```bash
# SLURM-specific configuration
SLURM_PARTITION=GPU-S          # Default partition
SLURM_QE_PATH=/cm/shared/apps/quantumespresso/qe-7.2/bin  # QE binary path
SLURM_DEFAULT_TIME=72:00:00    # Default time limit
SLURM_DEFAULT_MEMORY=32G       # Default memory allocation
```

### Customization Options

- **Partition Selection**: Choose appropriate partitions for different calculation types
- **Resource Allocation**: Optimize CPU, memory, and time limits
- **Module Loading**: Customize required software modules
- **Output Handling**: Configure output file naming and management

## 🔄 Workflow Integration

### With DFT Agent

The SLURM scheduler integrates seamlessly with the DFT agent:

```python
# Complete workflow example
workflow = {
    "1. Generate Structure": "dft_agent generates Pt(111) slab",
    "2. Create QE Input": "dft_agent creates scf.in file", 
    "3. Generate SLURM Script": "slurm_scheduler creates job script",
    "4. Submit Job": "slurm_scheduler submits to queue",
    "5. Monitor Progress": "slurm_scheduler tracks job status",
    "6. Retrieve Results": "dft_agent processes output"
}
```

### With Other Agents

- **Chatbot Agent**: Provides user-friendly job management interface
- **Database Agent**: Stores job metadata and results
- **Convergence Agent**: Manages parameter optimization workflows

## 📊 Job Tracking

### Workspace Organization

```
WORKSPACE/<thread_id>/
├── calculations/
│   ├── qe_inputs/          # QE input files
│   └── jobs/               # SLURM scripts and job info
│       ├── job_12345.json  # Job metadata
│       ├── job_12345.sh    # SLURM script
│       └── slurm-12345.out # Job output
└── results/                # Processed results
```

### Job Metadata

Each job is tracked with comprehensive metadata:

```json
{
  "job_id": "12345",
  "script_path": "/path/to/job.sh",
  "submitted_at": "2024-01-01T00:00:00",
  "status": "RUNNING",
  "thread_id": "calculation_001",
  "last_checked": "2024-01-01T01:00:00",
  "runtime": "01:00:00",
  "nodes": "node001"
}
```

## 🧪 Testing

### Running Tests

```bash
# Run SLURM scheduler tests
cd /home/ordillo/dft-agent
python tests/test_slurm_scheduler.py
```

### Test Coverage

- ✅ Script generation and customization
- ✅ Job submission and validation
- ✅ Status monitoring and tracking
- ✅ Error handling and recovery
- ✅ Integration with agent system
- ✅ Workspace management

## 🚨 Error Handling

### Common Issues and Solutions

1. **Job Submission Failures**
   - Invalid script syntax
   - Insufficient resources
   - Partition availability
   - File path issues

2. **Status Check Errors**
   - Network connectivity
   - SLURM service availability
   - Permission issues
   - Job ID validation

3. **Output Retrieval Problems**
   - File not found
   - Permission denied
   - Large file handling
   - Corrupted output

### Recovery Mechanisms

- Automatic retry for transient failures
- Graceful degradation for missing resources
- Comprehensive error logging
- User-friendly error messages

## 🔒 Security Considerations

### HPC Environment Safety

- **Resource Limits**: Prevents resource exhaustion
- **Queue Management**: Respects cluster policies
- **File Permissions**: Maintains proper access controls
- **Job Validation**: Prevents malicious script execution

### Best Practices

- Always validate job parameters before submission
- Use appropriate resource allocation
- Monitor job progress regularly
- Clean up completed jobs and temporary files
- Follow cluster usage policies

## 📈 Performance Optimization

### Resource Allocation

- **CPU Optimization**: Match tasks to available cores
- **Memory Management**: Allocate appropriate memory
- **Time Limits**: Set realistic time limits
- **Queue Selection**: Choose optimal partitions

### Monitoring Efficiency

- Batch status checks for multiple jobs
- Intelligent polling intervals
- Cached job information
- Efficient output processing

## 🔮 Future Enhancements

### Planned Features

- **Job Dependencies**: Handle job chains and workflows
- **Resource Prediction**: Intelligent resource allocation
- **Load Balancing**: Distribute jobs across partitions
- **Advanced Monitoring**: Real-time job visualization
- **Integration APIs**: Connect with external schedulers

### Extensibility

- Plugin architecture for custom schedulers
- Configurable job templates
- Custom monitoring dashboards
- Integration with workflow engines

## 📚 Examples

### Example 1: Simple SCF Calculation

```python
# Generate and submit a simple SCF job
result = scheduler.invoke({
    "messages": [HumanMessage(content="""
    Generate and submit a SLURM job for a Pt(111) slab SCF calculation.
    Use the input file 'pt_slab_scf.in' and allocate 16 cores for 2 hours.
    """)],
    "thread_id": "pt_slab_study"
})
```

### Example 2: Convergence Study

```python
# Submit multiple jobs for convergence testing
result = scheduler.invoke({
    "messages": [HumanMessage(content="""
    Submit jobs for cutoff energy convergence testing.
    Test cutoffs: 30, 40, 50, 60 Ry.
    Each job should run for 1 hour with 8 cores.
    """)],
    "thread_id": "convergence_study"
})
```

### Example 3: Job Monitoring

```python
# Monitor all jobs in a workspace
result = scheduler.invoke({
    "messages": [HumanMessage(content="""
    Check the status of all jobs in my workspace and provide a summary.
    """)],
    "thread_id": "my_workspace"
})
```

## 🤝 Contributing

### Development Guidelines

1. Follow existing code patterns and conventions
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure HPC environment compatibility
5. Test with real SLURM clusters when possible

### Code Structure

- Tools in `backend/agents/dft_tools/slurm_tools.py`
- Agent in `backend/agents/library/slurm_scheduler/`
- Tests in `tests/test_slurm_scheduler.py`
- Documentation in `docs/SLURM_SCHEDULER.md`

---

*The SLURM Job Scheduler Agent provides a robust, intelligent solution for HPC job management in computational materials science workflows. It seamlessly integrates with the DFT Agent ecosystem to provide end-to-end automation of complex calculations.*
