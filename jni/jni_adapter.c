#include <jni.h>
#include <stdlib.h>
#include <string.h>

/*
 * Minimal JNI helper functions for memory management
 * These bridge Java to the existing Rust C FFI
 */

// Allocate native memory
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

// Free native memory
JNIEXPORT void JNICALL
Java_com_example_embeddings_model_NativeMemory_freeNativeMemory(
    JNIEnv *env, jclass cls, jlong ptr)
{
    if (ptr != 0) {
        free((void *)ptr);
    }
}

// Copy Java byte array to native memory
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

// Write a long value to native memory
JNIEXPORT void JNICALL
Java_com_example_embeddings_model_NativeMemory_writeLong(
    JNIEnv *env, jclass cls, jlong ptr, jlong value)
{
    if (ptr != 0) {
        *((long *)ptr) = (long)value;
    }
}

// Read a long value from native memory
JNIEXPORT jlong JNICALL
Java_com_example_embeddings_model_NativeMemory_readLong(
    JNIEnv *env, jclass cls, jlong ptr)
{
    if (ptr == 0) {
        return 0;
    }
    return (jlong)(*((long *)ptr));
}

// Read float array from native memory
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

// Forward declarations for the existing Rust C FFI functions
extern int validate_embedding_method(const char *method);
extern int validate_embedding_model(int method_id, const char *model);
extern int generate_embeddings_from_texts(
    int method_id,
    int model_id,
    const void *inputs,
    size_t n_inputs,
    void *out_batch
);
extern void free_embedding_batch(void *batch);

// JNI wrappers for existing Rust functions

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

    int result = validate_embedding_model((int)methodId, model_str);

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

    int result = generate_embeddings_from_texts(
        (int)methodId,
        (int)modelId,
        (const void *)inputsPtr,
        (size_t)nInputs,
        (void *)outBatchPtr
    );

    return (jint)result;
}

JNIEXPORT void JNICALL
Java_com_example_embeddings_service_NativeBridge_freeEmbeddingBatch(
    JNIEnv *env, jclass cls, jlong batchPtr)
{
    if (batchPtr != 0) {
        free_embedding_batch((void *)batchPtr);
    }
}
