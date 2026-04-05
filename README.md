# Multiplicación de Matrices Paralela con OpenCL

## 📋 Descripción

Programa que implementa multiplicación de matrices utilizando **OpenCL** para paralelización en GPU o CPU. Prueba matrices de diferentes tamaños (512×512, 1024×1024, 2048×2048, 4096×4096) con medición de rendimiento en GFLOPS. Valida resultados comparando contra cálculos de referencia en CPU.

## 🎯 Funcionalidades

- Multiplicación de matrices 512×512 a 4096×4096 en paralelo
- Múltiples tamaños de prueba para análisis de escalabilidad
- Profiling de kernel OpenCL con medición de tiempo preciso
- Cálculo de rendimiento en GFLOPS (Gigas de operaciones por segundo)
- Validación automática con referencia de CPU
- Kernel OpenCL optimizado usando NDRange 2D
- Manejo robusto de errores
- Generador de números aleatorios para datos de prueba

## 🛠️ Requisitos

### En Fedora (entorno actual)
```bash
# Headers y librerías de OpenCL (compatibles con OpenCL-ICD-Loader)
sudo dnf install opencl-headers OpenCL-ICD-Loader-devel

# Compilador C++
sudo dnf install gcc-c++
```

### En Ubuntu/Debian
```bash
# Headers y librerías de OpenCL
sudo apt-get install opencl-headers ocl-icd-opencl-dev

# Compilador C++
sudo apt-get install build-essential
```

### En Windows
- Instalar drivers GPU (NVIDIA CUDA Toolkit, AMD ROCm, u OpenCL runtime correspondiente)
- Visual Studio o MinGW con soporte C++

## 🔧 Compilación

### Fedora/Linux 
```bash
g++ taller.cpp -o taller -lOpenCL

./taller
```

### Ubuntu/Linux
```bash
g++ taller.cpp -o taller -lOpenCL

./taller
```


### Windows (MinGW)
```bash
g++ -o taller.exe taller.cpp -lOpenCL
```

## ▶️ Ejecución

### Fedora/Linux 
```bash
./taller
```

### Ubuntu/Linux
```bash
./taller
```

### Windows
```bash
taller.exe
```

## 💻 Arquitectura del Código

### Componentes principales

1. **Kernel OpenCL** (`kernelSource`)
   - Función `matmul`: calcula cada elemento de la matriz resultante
   - Usa 2 dimensiones de threads (filas y columnas)

2. **Inicialización OpenCL**
   - Selecciona plataforma disponible
   - Busca dispositivos GPU/CPU
   - Crea contexto y comando queue con profiling habilitado

3. **Gestión de memoria**
   - Buffers GPU para matrices A, B y C (read-only, write-only)
   - Copia de datos A y B antes de ejecución

4. **Bucle de pruebas**
   - Itera sobre múltiples tamaños (512, 1024, 2048, 4096)
   - Genera datos aleatorios para cada tamaño
   - Calcula referencia de CPU para validación
   - Ejecuta kernel y mide tiempo con profiling

5. **Validación y medición**
   - Compara resultados GPU vs CPU (tolerancia 1e-3)
   - Calcula GFLOPS basado en tiempo del kernel
   - Reporta corrección de resultados

## 📊 Especificaciones

- **Tamaños de matrices**: 512×512, 1024×1024, 2048×2048, 4096×4096
- **Tipo de dato**: float (32 bits)
- **Tolerancia de validación**: 1e-3
- **Medición de tiempo**: Profiling de OpenCL (CL_PROFILING_COMMAND_START/END)
- **Métricas**: Tiempo en ms, GFLOPS (operaciones de punto flotante por segundo)
- **Verificación**: Comparación automática GPU vs CPU para cada tamaño

## 🐛 Solución de Problemas

### "No OpenCL platforms found"
- Instalar drivers GPU o librerías OpenCL para CPU
- En Ubuntu: `sudo apt-get install ocl-icd-libopencl1`

### "Failed to create context"
- Verificar que los drivers de GPU/CPU estén instalados
- Actualizar drivers GPU si es necesario

### Error de compilación "-lOpenCL no encontrado"
- Instalar: `sudo apt-get install ocl-icd-opencl-dev libopencl-clang-dev`

## 📝 Notas

- El programa prueba 4 diferentes tamaños de matrices en secuencia
- Cada iteración genera datos aleatorios independientes
- La validación CPU es necesaria pero puede ser lenta para matrices grandes
- El profiling de OpenCL requiere que el dispositivo soporte time_frequency
- Los GFLOPS calculados consideran 2 operaciones por multiplicación (suma y multiplicación)
- Libera todos los recursos OpenCL correctamente en cada iteración

## 👤 Autor

Taller de Computación Paralela

## 📚 Referencias

- [OpenCL Khronos Group](https://www.khronos.org/opencl/)
- [OpenCL Documentation](https://www.khronos.org/opencl/resources/)
