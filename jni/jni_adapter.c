#include <jni.h>
#include <stdlib.h>
#include <string.h>

/*
 * JNI adapter for Gembed Rust library
 * Bridges Java to the unified Rust C FFI
 */

// Input type constants (must match Rust library)
#define INPUT_TYPE_TEXT 0
#define INPUT_TYPE_IMAGE 1
#define INPUT_TYPE_MULTIMODAL 2

// Structure definitions matching the Rust C FFI
typedef struct {
    const char *ptr;
    size_t len;
} StringSlice;

typedef struct {
    const unsigned char *ptr;
    size_t len;
} ByteSlice;

typedef struct {
    int input_type;
    const ByteSlice *binary_data;
    size_t n_binary;
    const StringSlice *text_data;
    size_t n_text;
} InputData;

typedef struct {
    float *data;
    size_t n_vectors;
    size_t dim;
} EmbeddingBatch;

// Forward declarations for Rust C FFI functions
extern int validate_embedding_method(const char *method);
extern int validate_embedding_model(int method_id, const char *model, int input_type);
extern int generate_embeddings(
    int method_id,
    int model_id,
    const InputData *input_data,
    EmbeddingBatch *out_batch
);
extern void free_embedding_batch(EmbeddingBatch *batch);

// Memory Management Functions
JNIEXPORT jlong JNICALL
Java_com_example_embeddings_model_NativeMemory_allocateMemory(
    JNIEnv *env, jclass cls, jlong size)
{
    void *ptr = malloc((size_t)size);
    if (ptr == NULL) {
        return 0;
    }
    memset(ptr, 0, (size_t)size);
    return (jlong)ptr;
}

JNIEXPORT void JNICALL
Java_com_example_embeddings_model_NativeMemory_freeNativeMemory(
    JNIEnv *env, jclass cls, jlong ptr)
{
    if (ptr != 0) {
        free((void *)ptr);
    }
}

JNIEXPORT void JNICALL
Java_com_example_embeddings_model_NativeMemory_copyToNative(
    JNIEnv *env, jclass cls, jlong destPtr, jbyteArray src)
{
    if (destPtr == 0 || src == NULL) {
        return;
    }

    jsize len = (*env)->GetArrayLength(env, src);
    jbyte *bytes = (*env)->GetByteArrayElements(env, src, NULL);

    if (bytes != NULL) {
        memcpy((void *)destPtr, bytes, len);
        (*env)->ReleaseByteArrayElements(env, src, bytes, JNI_ABORT);
    }
}

JNIEXPORT void JNICALL
Java_com_example_embeddings_model_NativeMemory_writeLong(
    JNIEnv *env, jclass cls, jlong ptr, jlong value)
{
    if (ptr != 0) {
        *((long *)ptr) = (long)value;
    }
}

JNIEXPORT jlong JNICALL
Java_com_example_embeddings_model_NativeMemory_readLong(
    JNIEnv *env, jclass cls, jlong ptr)
{
    if (ptr == 0) {
        return 0;
    }
    return (jlong)(*((long *)ptr));
}

JNIEXPORT jfloatArray JNICALL
Java_com_example_embeddings_model_NativeMemory_readFloatArray(
    JNIEnv *env, jclass cls, jlong ptr, jlong count)
{
    if (ptr == 0 || count <= 0) {
        return NULL;
    }

    jfloatArray result = (*env)->NewFloatArray(env, (jsize)count);
    if (result == NULL) {
        return NULL;
    }

    (*env)->SetFloatArrayRegion(env, result, 0, (jsize)count, (jfloat *)ptr);

    return result;
}

// Embedding Generation Functions (New API)
JNIEXPORT jint JNICALL
Java_com_example_embeddings_service_NativeBridge_validateEmbeddingMethod(
    JNIEnv *env, jclass cls, jstring method)
{
    if (method == NULL) {
        return -1;
    }

    const char *method_str = (*env)->GetStringUTFChars(env, method, NULL);
    if (method_str == NULL) {
        return -1;
    }

    int result = validate_embedding_method(method_str);

    (*env)->ReleaseStringUTFChars(env, method, method_str);

    return (jint)result;
}

JNIEXPORT jint JNICALL
Java_com_example_embeddings_service_NativeBridge_validateEmbeddingModel(
    JNIEnv *env, jclass cls, jint methodId, jstring model)
{
    if (model == NULL) {
        return -1;
    }

    const char *model_str = (*env)->GetStringUTFChars(env, model, NULL);
    if (model_str == NULL) {
        return -1;
    }

    // Validate for text input type
    int result = validate_embedding_model((int)methodId, model_str, INPUT_TYPE_TEXT);

    (*env)->ReleaseStringUTFChars(env, model, model_str);

    return (jint)result;
}

JNIEXPORT jint JNICALL
Java_com_example_embeddings_service_NativeBridge_generateEmbeddingsFromTexts(
    JNIEnv *env, jclass cls,
    jint methodId, jint modelId,
    jlong inputsPtr, jint nInputs,
    jlong outBatchPtr)
{
    if (inputsPtr == 0 || outBatchPtr == 0) {
        return -1;
    }

    // Construct InputData structure for text inputs
    InputData input_data;
    input_data.input_type = INPUT_TYPE_TEXT;
    input_data.binary_data = NULL;
    input_data.n_binary = 0;
    input_data.text_data = (const StringSlice *)inputsPtr;
    input_data.n_text = (size_t)nInputs;

    // Call the unified generate_embeddings function
    int result = generate_embeddings(
        (int)methodId,
        (int)modelId,
        &input_data,
        (EmbeddingBatch *)outBatchPtr
    );

    return (jint)result;
}

JNIEXPORT void JNICALL
Java_com_example_embeddings_service_NativeBridge_freeEmbeddingBatch(
    JNIEnv *env, jclass cls, jlong batchPtr)
{
    if (batchPtr != 0) {
        free_embedding_batch((EmbeddingBatch *)batchPtr);
    }
}
