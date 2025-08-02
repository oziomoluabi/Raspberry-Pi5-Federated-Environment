"""
MATLAB Engine Integration for Python
Raspberry Pi 5 Federated Environmental Monitoring Network
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import structlog
import time
from pathlib import Path
import json
import threading
from dataclasses import dataclass

# Try to import MATLAB Engine
try:
    import matlab.engine
    import matlab
    MATLAB_AVAILABLE = True
except ImportError:
    MATLAB_AVAILABLE = False
    matlab = None

# Try to import Oct2Py as fallback
try:
    from oct2py import Oct2Py
    OCTAVE_AVAILABLE = True
except ImportError:
    OCTAVE_AVAILABLE = False
    Oct2Py = None

logger = structlog.get_logger(__name__)


@dataclass
class MATLABResult:
    """Data class for MATLAB computation results."""
    success: bool
    data: Any
    execution_time: float
    engine_type: str
    error_message: Optional[str] = None


class MATLABEngineManager:
    """Manager for MATLAB Engine API integration with Octave fallback."""
    
    def __init__(
        self,
        matlab_path: Optional[str] = None,
        prefer_matlab: bool = True,
        startup_timeout: float = 30.0
    ):
        """Initialize MATLAB Engine Manager."""
        
        self.matlab_path = matlab_path
        self.prefer_matlab = prefer_matlab
        self.startup_timeout = startup_timeout
        
        # Engine state
        self.matlab_engine = None
        self.octave_engine = None
        self.current_engine = None
        self.engine_type = None
        self.is_initialized = False
        
        # Performance tracking
        self.call_count = 0
        self.total_execution_time = 0.0
        self.last_call_time = None
        
        logger.info(
            "MATLAB Engine Manager initialized",
            matlab_available=MATLAB_AVAILABLE,
            octave_available=OCTAVE_AVAILABLE,
            prefer_matlab=prefer_matlab
        )
    
    def initialize_engines(self) -> bool:
        """Initialize MATLAB and/or Octave engines."""
        
        logger.info("Initializing computation engines...")
        
        # Try MATLAB first if preferred and available
        if self.prefer_matlab and MATLAB_AVAILABLE:
            if self._initialize_matlab():
                self.current_engine = self.matlab_engine
                self.engine_type = "matlab"
                self.is_initialized = True
                logger.info("MATLAB Engine initialized successfully")
                return True
        
        # Try Octave as fallback
        if OCTAVE_AVAILABLE:
            if self._initialize_octave():
                self.current_engine = self.octave_engine
                self.engine_type = "octave"
                self.is_initialized = True
                logger.info("Octave Engine initialized successfully")
                return True
        
        # If MATLAB wasn't preferred, try it now
        if not self.prefer_matlab and MATLAB_AVAILABLE:
            if self._initialize_matlab():
                self.current_engine = self.matlab_engine
                self.engine_type = "matlab"
                self.is_initialized = True
                logger.info("MATLAB Engine initialized as fallback")
                return True
        
        logger.error("Failed to initialize any computation engine")
        return False
    
    def _initialize_matlab(self) -> bool:
        """Initialize MATLAB Engine."""
        
        try:
            logger.info("Starting MATLAB Engine...")
            start_time = time.time()
            
            # Start MATLAB engine
            self.matlab_engine = matlab.engine.start_matlab()
            
            # Add MATLAB path if specified
            if self.matlab_path:
                matlab_path_obj = Path(self.matlab_path)
                if matlab_path_obj.exists():
                    self.matlab_engine.addpath(str(matlab_path_obj), nargout=0)
                    logger.info("Added MATLAB path", path=str(matlab_path_obj))
            
            # Test basic functionality
            result = self.matlab_engine.eval("2 + 2", nargout=1)
            if result != 4:
                raise RuntimeError("MATLAB engine test failed")
            
            startup_time = time.time() - start_time
            logger.info("MATLAB Engine started", startup_time=f"{startup_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize MATLAB Engine", error=str(e))
            self.matlab_engine = None
            return False
    
    def _initialize_octave(self) -> bool:
        """Initialize Octave Engine."""
        
        try:
            logger.info("Starting Octave Engine...")
            start_time = time.time()
            
            # Start Octave engine
            self.octave_engine = Oct2Py()
            
            # Add Octave path if specified
            if self.matlab_path:
                octave_path_obj = Path(self.matlab_path)
                if octave_path_obj.exists():
                    self.octave_engine.addpath(str(octave_path_obj))
                    logger.info("Added Octave path", path=str(octave_path_obj))
            
            # Test basic functionality
            result = self.octave_engine.eval("2 + 2")
            if result != 4:
                raise RuntimeError("Octave engine test failed")
            
            startup_time = time.time() - start_time
            logger.info("Octave Engine started", startup_time=f"{startup_time:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error("Failed to initialize Octave Engine", error=str(e))
            self.octave_engine = None
            return False
    
    def call_function(
        self,
        function_name: str,
        *args,
        nargout: int = 1,
        **kwargs
    ) -> MATLABResult:
        """Call MATLAB/Octave function with arguments."""
        
        if not self.is_initialized:
            if not self.initialize_engines():
                return MATLABResult(
                    success=False,
                    data=None,
                    execution_time=0.0,
                    engine_type="none",
                    error_message="No computation engine available"
                )
        
        start_time = time.time()
        
        try:
            logger.debug(
                "Calling function",
                function=function_name,
                engine=self.engine_type,
                args_count=len(args),
                nargout=nargout
            )
            
            if self.engine_type == "matlab":
                result = self._call_matlab_function(function_name, *args, nargout=nargout, **kwargs)
            elif self.engine_type == "octave":
                result = self._call_octave_function(function_name, *args, nargout=nargout, **kwargs)
            else:
                raise RuntimeError("No valid engine available")
            
            execution_time = time.time() - start_time
            
            # Update performance tracking
            self.call_count += 1
            self.total_execution_time += execution_time
            self.last_call_time = time.time()
            
            logger.debug(
                "Function call completed",
                function=function_name,
                execution_time=f"{execution_time:.3f}s",
                engine=self.engine_type
            )
            
            return MATLABResult(
                success=True,
                data=result,
                execution_time=execution_time,
                engine_type=self.engine_type
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Function call failed: {str(e)}"
            
            logger.error(
                "Function call failed",
                function=function_name,
                engine=self.engine_type,
                error=str(e),
                execution_time=f"{execution_time:.3f}s"
            )
            
            return MATLABResult(
                success=False,
                data=None,
                execution_time=execution_time,
                engine_type=self.engine_type,
                error_message=error_msg
            )
    
    def _call_matlab_function(self, function_name: str, *args, nargout: int = 1, **kwargs) -> Any:
        """Call MATLAB function using MATLAB Engine."""
        
        # Convert numpy arrays to MATLAB arrays
        matlab_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                if arg.dtype == np.float64:
                    matlab_args.append(matlab.double(arg.tolist()))
                elif arg.dtype == np.float32:
                    matlab_args.append(matlab.single(arg.tolist()))
                elif arg.dtype in [np.int32, np.int64]:
                    matlab_args.append(matlab.int32(arg.tolist()))
                else:
                    matlab_args.append(matlab.double(arg.astype(float).tolist()))
            elif isinstance(arg, (list, tuple)):
                matlab_args.append(matlab.double(arg))
            else:
                matlab_args.append(arg)
        
        # Get function handle
        func = getattr(self.matlab_engine, function_name)
        
        # Call function
        if nargout == 1:
            result = func(*matlab_args, **kwargs)
        else:
            result = func(*matlab_args, nargout=nargout, **kwargs)
        
        # Convert MATLAB arrays back to numpy
        if hasattr(result, '_data'):
            return np.array(result._data).reshape(result.size)
        elif isinstance(result, (list, tuple)) and len(result) > 0:
            converted_result = []
            for item in result:
                if hasattr(item, '_data'):
                    converted_result.append(np.array(item._data).reshape(item.size))
                else:
                    converted_result.append(item)
            return converted_result if len(converted_result) > 1 else converted_result[0]
        else:
            return result
    
    def _call_octave_function(self, function_name: str, *args, nargout: int = 1, **kwargs) -> Any:
        """Call Octave function using Oct2Py."""
        
        # Oct2Py handles numpy arrays automatically
        func = getattr(self.octave_engine, function_name)
        
        if nargout == 1:
            result = func(*args, **kwargs)
        else:
            result = func(*args, nout=nargout, **kwargs)
        
        return result
    
    def evaluate_expression(self, expression: str) -> MATLABResult:
        """Evaluate MATLAB/Octave expression."""
        
        if not self.is_initialized:
            if not self.initialize_engines():
                return MATLABResult(
                    success=False,
                    data=None,
                    execution_time=0.0,
                    engine_type="none",
                    error_message="No computation engine available"
                )
        
        start_time = time.time()
        
        try:
            logger.debug("Evaluating expression", expression=expression, engine=self.engine_type)
            
            if self.engine_type == "matlab":
                result = self.matlab_engine.eval(expression, nargout=1)
            elif self.engine_type == "octave":
                result = self.octave_engine.eval(expression)
            else:
                raise RuntimeError("No valid engine available")
            
            execution_time = time.time() - start_time
            
            logger.debug(
                "Expression evaluated",
                expression=expression,
                execution_time=f"{execution_time:.3f}s",
                engine=self.engine_type
            )
            
            return MATLABResult(
                success=True,
                data=result,
                execution_time=execution_time,
                engine_type=self.engine_type
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Expression evaluation failed: {str(e)}"
            
            logger.error(
                "Expression evaluation failed",
                expression=expression,
                engine=self.engine_type,
                error=str(e)
            )
            
            return MATLABResult(
                success=False,
                data=None,
                execution_time=execution_time,
                engine_type=self.engine_type,
                error_message=error_msg
            )
    
    def set_variable(self, name: str, value: Any) -> bool:
        """Set variable in MATLAB/Octave workspace."""
        
        if not self.is_initialized:
            return False
        
        try:
            if self.engine_type == "matlab":
                # Convert numpy arrays to MATLAB format
                if isinstance(value, np.ndarray):
                    if value.dtype == np.float64:
                        matlab_value = matlab.double(value.tolist())
                    elif value.dtype == np.float32:
                        matlab_value = matlab.single(value.tolist())
                    else:
                        matlab_value = matlab.double(value.astype(float).tolist())
                else:
                    matlab_value = value
                
                self.matlab_engine.workspace[name] = matlab_value
                
            elif self.engine_type == "octave":
                # Oct2Py handles conversion automatically
                setattr(self.octave_engine, name, value)
            
            logger.debug("Variable set", name=name, engine=self.engine_type)
            return True
            
        except Exception as e:
            logger.error("Failed to set variable", name=name, error=str(e))
            return False
    
    def get_variable(self, name: str) -> Optional[Any]:
        """Get variable from MATLAB/Octave workspace."""
        
        if not self.is_initialized:
            return None
        
        try:
            if self.engine_type == "matlab":
                result = self.matlab_engine.workspace[name]
                # Convert MATLAB arrays to numpy
                if hasattr(result, '_data'):
                    return np.array(result._data).reshape(result.size)
                else:
                    return result
                    
            elif self.engine_type == "octave":
                return getattr(self.octave_engine, name)
            
        except Exception as e:
            logger.error("Failed to get variable", name=name, error=str(e))
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        
        if self.call_count > 0:
            avg_execution_time = self.total_execution_time / self.call_count
        else:
            avg_execution_time = 0.0
        
        return {
            'engine_type': self.engine_type,
            'is_initialized': self.is_initialized,
            'call_count': self.call_count,
            'total_execution_time': self.total_execution_time,
            'avg_execution_time': avg_execution_time,
            'last_call_time': self.last_call_time,
            'matlab_available': MATLAB_AVAILABLE,
            'octave_available': OCTAVE_AVAILABLE
        }
    
    def shutdown(self) -> None:
        """Shutdown engines and cleanup."""
        
        logger.info("Shutting down computation engines")
        
        try:
            if self.matlab_engine:
                self.matlab_engine.quit()
                self.matlab_engine = None
                logger.info("MATLAB Engine shut down")
        except Exception as e:
            logger.error("Error shutting down MATLAB Engine", error=str(e))
        
        try:
            if self.octave_engine:
                self.octave_engine.exit()
                self.octave_engine = None
                logger.info("Octave Engine shut down")
        except Exception as e:
            logger.error("Error shutting down Octave Engine", error=str(e))
        
        self.current_engine = None
        self.engine_type = None
        self.is_initialized = False


class EnvironmentalDataProcessor:
    """High-level interface for environmental data processing using MATLAB."""
    
    def __init__(self, matlab_manager: MATLABEngineManager):
        """Initialize environmental data processor."""
        self.matlab_manager = matlab_manager
        
        logger.info("Environmental data processor initialized")
    
    def process_environmental_data(
        self,
        temperature: np.ndarray,
        humidity: np.ndarray,
        timestamps: np.ndarray,
        options: Optional[Dict] = None
    ) -> Dict:
        """Process environmental data using MATLAB env_preprocess function."""
        
        logger.info(
            "Processing environmental data",
            temp_samples=len(temperature),
            humidity_samples=len(humidity),
            time_span=f"{(timestamps[-1] - timestamps[0]) / 3600:.2f} hours"
        )
        
        # Prepare data structure for MATLAB
        raw_data = {
            'temperature': temperature,
            'humidity': humidity,
            'timestamp': timestamps
        }
        
        # Set variables in MATLAB workspace
        self.matlab_manager.set_variable('raw_data', raw_data)
        if options:
            self.matlab_manager.set_variable('options', options)
        
        # Call MATLAB preprocessing function
        if options:
            result = self.matlab_manager.call_function(
                'env_preprocess',
                'raw_data', 'options',
                nargout=3
            )
        else:
            result = self.matlab_manager.call_function(
                'env_preprocess',
                'raw_data',
                nargout=3
            )
        
        if not result.success:
            logger.error("Environmental data processing failed", error=result.error_message)
            return {'success': False, 'error': result.error_message}
        
        # Extract results
        processed_data, stats, forecast = result.data
        
        logger.info(
            "Environmental data processing completed",
            execution_time=f"{result.execution_time:.3f}s",
            engine=result.engine_type
        )
        
        return {
            'success': True,
            'processed_data': processed_data,
            'statistics': stats,
            'forecast': forecast,
            'execution_time': result.execution_time,
            'engine_type': result.engine_type
        }


class SimulinkModelRunner:
    """Interface for running Simulink models from Python."""
    
    def __init__(self, matlab_manager: MATLABEngineManager):
        """Initialize Simulink model runner."""
        self.matlab_manager = matlab_manager
        
        logger.info("Simulink model runner initialized")
    
    def run_predictive_maintenance_model(
        self,
        vibration_data: np.ndarray,
        temperature_data: np.ndarray,
        humidity_data: np.ndarray,
        simulation_time: float = 100.0
    ) -> Dict:
        """Run predictive maintenance Simulink model."""
        
        logger.info(
            "Running predictive maintenance model",
            vibration_samples=len(vibration_data),
            env_samples=len(temperature_data),
            sim_time=simulation_time
        )
        
        # Prepare time vectors
        t_vib = np.linspace(0, simulation_time, len(vibration_data))
        t_env = np.linspace(0, simulation_time, len(temperature_data))
        
        # Create timeseries data for Simulink
        self.matlab_manager.evaluate_expression(
            f"vibration_data = timeseries({vibration_data.tolist()}, {t_vib.tolist()})"
        )
        self.matlab_manager.evaluate_expression(
            f"temperature_data = timeseries({temperature_data.tolist()}, {t_env.tolist()})"
        )
        self.matlab_manager.evaluate_expression(
            f"humidity_data = timeseries({humidity_data.tolist()}, {t_env.tolist()})"
        )
        
        # Run simulation
        result = self.matlab_manager.call_function(
            'sim',
            'predictive_maintenance',
            nargout=1
        )
        
        if not result.success:
            logger.error("Simulink simulation failed", error=result.error_message)
            return {'success': False, 'error': result.error_message}
        
        # Extract results from workspace
        health_score_log = self.matlab_manager.get_variable('health_score_log')
        anomaly_log = self.matlab_manager.get_variable('anomaly_log')
        
        logger.info(
            "Simulink simulation completed",
            execution_time=f"{result.execution_time:.3f}s",
            engine=result.engine_type
        )
        
        return {
            'success': True,
            'simulation_output': result.data,
            'health_score_log': health_score_log,
            'anomaly_log': anomaly_log,
            'execution_time': result.execution_time,
            'engine_type': result.engine_type
        }


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Initialize MATLAB manager
    manager = MATLABEngineManager(
        matlab_path=str(Path(__file__).parent.parent.parent / "matlab"),
        prefer_matlab=True
    )
    
    if manager.initialize_engines():
        print(f"Using {manager.engine_type} engine")
        
        # Test basic functionality
        result = manager.evaluate_expression("sqrt(16)")
        print(f"sqrt(16) = {result.data} (took {result.execution_time:.3f}s)")
        
        # Test environmental data processing
        processor = EnvironmentalDataProcessor(manager)
        
        # Generate test data
        t = np.linspace(0, 24*3600, 100)  # 24 hours
        temp = 25 + 5 * np.sin(2*np.pi*t/(24*3600)) + np.random.normal(0, 1, len(t))
        humidity = 60 - 10 * np.sin(2*np.pi*t/(24*3600)) + np.random.normal(0, 2, len(t))
        
        # Process data
        env_result = processor.process_environmental_data(temp, humidity, t)
        
        if env_result['success']:
            print("Environmental data processing successful!")
            print(f"Execution time: {env_result['execution_time']:.3f}s")
        else:
            print(f"Environmental data processing failed: {env_result['error']}")
        
        # Shutdown
        manager.shutdown()
    else:
        print("Failed to initialize any computation engine")
