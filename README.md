# Multiplicación de Matrices Paralela con OpenCL

## 📋 Descripción

Programa que implementa multiplicación de matrices 256×256 utilizando **OpenCL** para paralelización en GPU o CPU. El programa automáticamente detecta dispositivos disponibles y elige el mejor (GPU si está disponible, sino CPU).

## 🎯 Funcionalidades

- Multiplicación de matrices 256×256 en paralelo
- Detección automática de dispositivos OpenCL (GPU/CPU)
- Kernel OpenCL optimizado usando NDRange 2D
- Manejo robusto de errores
- Generador de números aleatorios para datos de prueba

## 🛠️ Requisitos

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

### Ubuntu/Linux
```bash
g++ -o taller taller -lOpenCL
```

### Windows (MinGW)
```bash
g++ -o taller.exe taller -lOpenCL
```

## ▶️ Ejecución

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

1. **Kernel OpenCL** (`kernel_src`)
   - Función `matmul`: calcula cada elemento de la matriz resultante
   - Usa 2 dimensiones de threads (filas y columnas)

2. **Inicialización OpenCL**
   - Selecciona plataforma disponible
   - Busca dispositivos GPU/CPU
   - Crea contexto y comando queue

3. **Gestión de memoria**
   - Buffers GPU para matrices A, B y C (read-only, write-only)
   - Copia de datos precálculados (256×256 floats)

4. **Ejecución**
   - Compila kernel en tiempo de ejecución
   - Encolola kernel para ejecución paralela
   - Lee resultado de vuelta a CPU

## 📊 Especificaciones

- **Tamaño matriz**: 256×256 floats
- **Memoria**: ~0.75 MB por matriz (total 2.25 MB)
- **Threads**: 256×256 = 65,536 threads paralelos
- **Tipo de dato**: float (32 bits)

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

- El programa usa valores aleatorios con seed fija (123) para reproducibilidad
- Assertions verifican que las dimensiones sean correctas
- Libera todos los recursos OpenCL adecuadamente

## 👤 Autor

Taller de Computación Paralela

## 📚 Referencias

- [OpenCL Khronos Group](https://www.khronos.org/opencl/)
- [OpenCL Documentation](https://www.khronos.org/opencl/resources/)
