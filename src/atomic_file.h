#ifndef ATOMIC_FILE_H
#define ATOMIC_FILE_H

#include <stdio.h>

/* Write a sibling temporary file, then publish it with one rename. Until that
 * rename succeeds, the destination (including an earlier checkpoint) is intact.
 * The caller keeps destination_path alive until commit or discard. */
typedef struct {
    FILE *stream;
    char *temporary_path;
    const char *destination_path;
} AtomicFile;

int open_atomic_file(AtomicFile *file, const char *destination_path);
int commit_atomic_file(AtomicFile *file);
void discard_atomic_file(AtomicFile *file);

#endif
