#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "atomic_file.h"

int open_atomic_file(AtomicFile *file, const char *destination_path) {
    *file = (AtomicFile){0};
    const char suffix[] = ".tmp.XXXXXX";
    size_t path_length = strlen(destination_path);
    if (path_length > SIZE_MAX - sizeof(suffix)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    file->temporary_path = malloc(path_length + sizeof(suffix));
    if (!file->temporary_path) {
        return -1;
    }
    memcpy(file->temporary_path, destination_path, path_length);
    memcpy(file->temporary_path + path_length, suffix, sizeof(suffix));
    file->destination_path = destination_path;

    /* mkstemp creates a unique file exclusively; a fixed .tmp name would let
     * simultaneous writers overwrite one another's unfinished output. */
    int descriptor = mkstemp(file->temporary_path);
    if (descriptor < 0) {
        int saved_errno = errno;
        free(file->temporary_path);
        *file = (AtomicFile){0};
        errno = saved_errno;
        return -1;
    }
    file->stream = fdopen(descriptor, "wb");
    if (!file->stream) {
        int saved_errno = errno;
        close(descriptor);
        discard_atomic_file(file);
        errno = saved_errno;
        return -1;
    }
    return 0;
}

void discard_atomic_file(AtomicFile *file) {
    if (file->stream) {
        fclose(file->stream);
    }
    if (file->temporary_path) {
        remove(file->temporary_path);
        free(file->temporary_path);
    }
    *file = (AtomicFile){0};
}

int commit_atomic_file(AtomicFile *file) {
    /* fclose flushes buffered writes and can report a late disk-full error.
     * Only a fully written, successfully closed file may replace the target. */
    int status = fclose(file->stream);
    file->stream = NULL;
    if (status == 0) {
        status = rename(file->temporary_path, file->destination_path);
    }
    if (status != 0) {
        int saved_errno = errno;
        discard_atomic_file(file);
        errno = saved_errno;
        return -1;
    }
    free(file->temporary_path);
    *file = (AtomicFile){0};
    return 0;
}
