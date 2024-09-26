#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC extern
#endif

#ifdef __GNUC__
#define EXPORT EXTERNC __attribute__((visibility("default")))
#elif defined(_MSC_VER)
#define EXPORT EXTERNC __attribute__ ((dllexport))
#else
#define EXPORT EXTERNC
#endif

EXPORT int RtmStreamStart(void **const handle, const int devnum, const int NBUFS, int *const maxlen);
EXPORT int RtmStreamStop(void *const handle);
EXPORT int RtmStreamClose(void *const handle);
EXPORT int RtmStreamGetBuffer(void *const handle, void *const buf, const int buflen);

#undef EXPORT
#define EXPORT EXTERNC
