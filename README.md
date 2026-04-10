# Spring Gembed Demo

A Spring Boot application that demonstrates **in-process embedding generation** from Java, by using the
[Gembed Rust core](https://github.com/JoelDiaz222/gembed).

The application bridges the JVM to the portable Gembed Rust core (`libgembed`) via JNI. A thin C adapter
(`jni_adapter.c`) translates JNI types into the C ABI expected by the Rust library, enabling embedding generation.

## Architecture

```
┌──────────────────────────────────────────────────┐
│           Spring Boot REST API                   │
│   POST /embed { backend, model, texts }          │
└──────────────────────┬───────────────────────────┘
                       │  Java Service
                       ▼
┌──────────────────────────────────────────────────┐
│           EmbeddingService (Java)                │
│  - Calls NativeBridge (native methods)           │
│  - Manages native memory for StringSlice arrays  │
└──────────────────────┬───────────────────────────┘
                       │  JNI (Java Native Interface)
                       ▼
┌──────────────────────────────────────────────────┐
│           JNI Adapter (jni_adapter.c)            │
│  - Marshals JNI types → C ABI types              │
│  - Calls validate_backend / generate_embeddings  │
└──────────────────────┬───────────────────────────┘
                       │  C FFI
                       ▼
┌──────────────────────────────────────────────────┐
│        Rust Core Library (libgembed_jni.so)      │
│  Backends: embed_anything / FastEmbed /          │
│            ORT / gRPC / HTTP                     │
└──────────────────────────────────────────────────┘
```

## Requirements

- Java 23+
- Rust toolchain (`cargo`)
- `gcc` / `clang`

## Build & Run

The Gradle build automatically compiles the JNI adapter and Rust core before building the Java sources.

```bash
git clone --recurse-submodules https://github.com/JoelDiaz222/spring-gembed-demo
cd spring-gembed-demo

./gradlew bootRun
```

The server starts on `http://localhost:8080`.

## API

### `POST /embed`

Generate embeddings for a batch of texts.

**Request:**

```json
{
  "backend": "embed_anything",
  "model": "Qdrant/all-MiniLM-L6-v2-onnx",
  "texts": [
    "Hello world",
    "Embedding from Java"
  ]
}
```

**Response:**

```json
{
  "backend": "embed_anything",
  "model": "Qdrant/all-MiniLM-L6-v2-onnx",
  "embeddings": [
    [
      0.12,
      -0.34,
      ...
    ],
    [
      0.56,
      0.78,
      ...
    ]
  ]
}
```

## License

Licensed under the [Apache License 2.0](./LICENSE).
