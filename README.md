# Unified Async LLM Client

Pre-entrega 1 del curso de AI Engineering.

Este proyecto implementa un cliente unificado y asíncrono para trabajar con múltiples proveedores de modelos de lenguaje bajo una misma interfaz.

La implementación incluye soporte para:

* OpenAI
* Anthropic
* Google Gemini

OpenAI y Anthropic forman parte de los requisitos principales de la entrega. Gemini se incorpora como proveedor adicional para demostrar la extensibilidad de la arquitectura y permitir realizar pruebas reales utilizando su API.

## Objetivos cumplidos

* Interfaz común para OpenAI, Anthropic y Gemini.
* Uso exclusivo de clientes asíncronos.
* Generación normal mediante `async/await`.
* Streaming mediante generadores asíncronos y `yield`.
* Validación con Pydantic.
* API keys almacenadas como `SecretStr`.
* Validación de `temperature` entre `0` y `2`.
* Validación de `max_tokens`.
* Selección dinámica del proveedor mediante `LLM_PROVIDER`.
* Manejo controlado de errores de conexión, timeout, rate limit y errores HTTP.
* Retries configurables según las capacidades de los SDK oficiales.
* Variables de entorno mediante `python-dotenv`.
* Intercambiabilidad de proveedores sin modificar la lógica principal de la aplicación.

## Estructura

```text
unified-async-llm-client/
├── llm_client/
│   ├── __init__.py
│   ├── schemas.py
│   ├── base.py
│   ├── factory.py
│   ├── manager.py
│   ├── openai_client.py
│   ├── anthropic_client.py
│   └── gemini_client.py
├── .env.example
├── .gitignore
├── main.py
├── requirements.txt
└── README.md
```

## Requisitos

* Python 3.12+
* Una API key de al menos uno de los proveedores configurados:

  * OpenAI
  * Anthropic
  * Google Gemini

Para las pruebas realizadas en esta pre-entrega se utilizó Google Gemini.

## 1. Crear el entorno virtual

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Para verificar la versión activa:

```bash
python --version
```

Debe utilizar Python 3.12 o superior.

## 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Dependencias principales:

* `openai`
* `anthropic`
* `google-genai`
* `pydantic`
* `python-dotenv`

## 3. Configurar variables de entorno

Copiar `.env.example` como `.env`.

### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

### Linux / macOS

```bash
cp .env.example .env
```

El archivo `.env` contiene las credenciales reales y no debe subirse al repositorio.

### Ejemplo con Gemini

```env
LLM_PROVIDER=gemini

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=tu_api_key

OPENAI_MODEL=gpt-4.1-mini
ANTHROPIC_MODEL=claude-sonnet-4-6
GEMINI_MODEL=gemini-3.5-flash-lite

TEMPERATURE=0.2
MAX_TOKENS=300
REQUEST_TIMEOUT=60
MAX_RETRIES=2
```

### Ejemplo con OpenAI

```env
LLM_PROVIDER=openai

OPENAI_API_KEY=tu_api_key
ANTHROPIC_API_KEY=
GEMINI_API_KEY=

OPENAI_MODEL=gpt-4.1-mini
ANTHROPIC_MODEL=claude-sonnet-4-6
GEMINI_MODEL=gemini-3.5-flash-lite

TEMPERATURE=0.2
MAX_TOKENS=300
REQUEST_TIMEOUT=30
MAX_RETRIES=2
```

### Ejemplo con Anthropic

```env
LLM_PROVIDER=anthropic

OPENAI_API_KEY=
ANTHROPIC_API_KEY=tu_api_key
GEMINI_API_KEY=

OPENAI_MODEL=gpt-4.1-mini
ANTHROPIC_MODEL=claude-sonnet-4-6
GEMINI_MODEL=gemini-3.5-flash-lite

TEMPERATURE=0.2
MAX_TOKENS=300
REQUEST_TIMEOUT=30
MAX_RETRIES=2
```

El archivo `.env` está excluido mediante `.gitignore`.

El archivo `.env.example` sí forma parte del repositorio, pero nunca debe contener API keys reales.

## 4. Ejecutar

```bash
python main.py
```

El script realiza la pregunta:

> ¿Qué es la entropía?

Primero obtiene una respuesta completa y después realiza la misma consulta en modo streaming.

Ejemplo de ejecución con Gemini:

```text
Proveedor: gemini | Modelo: gemini-3.5-flash-lite

=== MODO NORMAL ===

La entropía es...

=== MODO STREAMING ===

La entropía es...
```

## Arquitectura

La aplicación sigue una arquitectura orientada a interfaces y desacoplada del proveedor concreto.

```text
main.py
   │
   ▼
AsyncLLMManager
   │
   ▼
LLMFactory
   │
   ├── OpenAIClient
   ├── AnthropicClient
   └── GeminiClient
          │
          ▼
     Provider API
```

La lógica principal utiliza siempre la misma interfaz independientemente del proveedor seleccionado.

## `schemas.py`

Contiene los modelos Pydantic utilizados para validar configuración, mensajes y respuestas.

### `ChatMessage`

Normaliza los mensajes de entrada.

```python
ChatMessage(
    role="user",
    content="¿Qué es la entropía?"
)
```

Los roles soportados son:

* `system`
* `user`
* `assistant`

### `ModelParameters`

Valida los parámetros comunes de generación.

```python
ModelParameters(
    temperature=0.2,
    max_tokens=300,
    timeout_seconds=30,
)
```

Validaciones principales:

* `temperature`: entre `0` y `2`.
* `max_tokens`: mayor a `0`.
* `timeout_seconds`: mayor a `0`.

### `LLMSettings`

Carga:

* proveedor seleccionado;
* modelos;
* API keys;
* configuración de retries.

Las API keys se almacenan mediante `SecretStr` para reducir el riesgo de exposición accidental.

### `ModelResponse`

Normaliza respuestas exitosas y errores.

Ejemplo:

```python
ModelResponse(
    success=True,
    provider="gemini",
    model="gemini-3.5-flash-lite",
    content="..."
)
```

También puede representar errores controlados:

```python
ModelResponse(
    success=False,
    provider="gemini",
    model="gemini-3.5-flash-lite",
    error="Rate limit alcanzado..."
)
```

## `BaseLLMClient`

Define el contrato común que deben implementar todos los proveedores.

```python
async def generate(...)

async def stream(...)
```

Esto permite desacoplar la aplicación de los SDK específicos.

## `OpenAIClient`

Implementa `BaseLLMClient` utilizando el cliente asíncrono oficial de OpenAI.

Características:

* generación asíncrona;
* streaming;
* timeout;
* manejo de errores HTTP;
* manejo de rate limits;
* retries configurables.

## `AnthropicClient`

Implementa la misma interfaz utilizando `AsyncAnthropic`.

La aplicación puede cambiar de OpenAI a Anthropic modificando únicamente:

```env
LLM_PROVIDER=anthropic
```

sin cambiar `main.py`.

## `GeminiClient`

Implementa `BaseLLMClient` utilizando el SDK oficial `google-genai`.

El cliente utiliza la API asíncrona:

```python
client.aio
```

para evitar bloquear el event loop.

También implementa streaming mediante un generador asíncrono.

Gemini fue utilizado para ejecutar y validar el proyecto de forma real durante el desarrollo de esta pre-entrega.

## `LLMFactory`

Centraliza la creación de clientes según el valor de:

```env
LLM_PROVIDER
```

Por ejemplo:

```env
LLM_PROVIDER=openai
```

crea un `OpenAIClient`.

```env
LLM_PROVIDER=anthropic
```

crea un `AnthropicClient`.

```env
LLM_PROVIDER=gemini
```

crea un `GeminiClient`.

Esto evita condicionales relacionados con proveedores distribuidos por toda la aplicación.

## `AsyncLLMManager`

Actúa como fachada para la aplicación.

Generación normal:

```python
response = await manager.generate(messages)
```

Streaming:

```python
async for chunk in manager.stream(messages):
    print(chunk, end="", flush=True)
```

El código consumidor permanece igual independientemente del proveedor seleccionado.

## Streaming

Uno de los principales objetivos de la implementación es permitir que la respuesta pueda entregarse progresivamente.

El contrato utiliza un `AsyncIterator` y `yield`.

Ejemplo:

```python
async for chunk in manager.stream(messages):
    print(chunk, end="", flush=True)
```

Esto permite comenzar a mostrar la respuesta mientras todavía está siendo generada por el modelo.

## Asincronía

Todos los accesos a APIs externas se realizan utilizando clientes asíncronos.

No se utilizan:

```python
time.sleep()
```

ni clientes HTTP síncronos dentro de las corrutinas principales.

El punto de entrada utiliza:

```python
if __name__ == "__main__":
    asyncio.run(main())
```

## Manejo de errores

La aplicación captura errores relacionados con:

* rate limiting;
* timeout;
* conexión;
* estados HTTP;
* indisponibilidad temporal del proveedor;
* errores inesperados del SDK.

En modo normal, en lugar de producir un crash, se devuelve un `ModelResponse` controlado.

Por ejemplo:

```text
[ERROR] Gemini respondió con error HTTP 503
```

En streaming, los errores son entregados como fragmentos controlados:

```text
[ERROR] La solicitud a Gemini superó el tiempo de espera.
```

Durante las pruebas con Gemini se validó también este comportamiento cuando un modelo se encontraba temporalmente bajo alta demanda.

## Retries

`MAX_RETRIES` permite configurar la estrategia de reintentos disponible en los SDK que la soportan.

Ejemplo:

```env
MAX_RETRIES=2
```

Esto ayuda a tolerar errores transitorios como:

* `429 Too Many Requests`;
* errores `5xx`;
* fallos temporales de red.

## Consideraciones sobre `temperature`

La consigna solicita validar `temperature` entre `0` y `2`.

Por ese motivo se mantiene como parámetro común en `ModelParameters`:

```python
temperature: float = Field(
    default=0.2,
    ge=0.0,
    le=2.0
)
```

Sin embargo, no todos los proveedores o modelos soportan exactamente los mismos parámetros.

La arquitectura permite que cada cliente adapte la configuración común a las capacidades particulares de su proveedor.

Por ejemplo, algunos modelos recientes de Gemini pueden no utilizar `temperature`, aunque el parámetro siga siendo validado como parte del esquema común solicitado por la consigna.

## Seguridad

Las API keys:

* se cargan desde `.env`;
* no están hardcodeadas;
* usan `SecretStr`;
* no se incluyen en logs;
* `.env` está ignorado mediante `.gitignore`.

El repositorio incluye únicamente:

```env
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
```

en `.env.example`.

Nunca deben agregarse credenciales reales a `.env.example` ni a ningún archivo versionado por Git.

## Criterios de la entrega

| Criterio                   | Implementación            |
| -------------------------- | ------------------------- |
| Python 3.12                | Sí                        |
| OpenAI                     | Sí                        |
| Anthropic                  | Sí                        |
| Gemini                     | Sí, proveedor adicional   |
| Interfaz común             | `BaseLLMClient`           |
| Manager unificado          | `AsyncLLMManager`         |
| Factory                    | `LLMFactory`              |
| Async/await                | Sí                        |
| Streaming                  | `AsyncIterator` + `yield` |
| Pydantic                   | `schemas.py`              |
| Temperature 0-2            | `Field(ge=0, le=2)`       |
| `max_tokens`               | Validado                  |
| `.env.example`             | Sí                        |
| `SecretStr`                | Sí                        |
| Rate limit controlado      | Sí                        |
| Timeout controlado         | Sí                        |
| Errores de red controlados | Sí                        |
| Retry                      | `MAX_RETRIES`             |
| OpenAI async               | Sí                        |
| Anthropic async            | Sí                        |
| Gemini async               | Sí                        |
| Script de validación       | `main.py`                 |
| Generación normal          | Sí                        |
| Streaming real             | Sí                        |

## Resultado

El proyecto permite cambiar de proveedor únicamente modificando una variable de entorno:

```env
LLM_PROVIDER=openai
```

```env
LLM_PROVIDER=anthropic
```

o:

```env
LLM_PROVIDER=gemini
```

sin modificar la lógica de negocio ni el código de `main.py`.

Esto demuestra la intercambiabilidad buscada por el ejercicio y deja una base extensible para incorporar nuevos proveedores en el futuro.
