
#ifndef _LOCAL_H_
#define _LOCAL_H_


/*
 * String Handling
 */

#ifndef __KERNEL__
#include <string.h>
#endif

#define STREQ( s1, s2 )        (strcmp( s1, s2 ) == 0)
#define STREQN( s1, s2, n )    (strncmp( s1, s2, n ) == 0) 
#define STREQNL( s1, s2 )      (strncmp( s1, s2, strlen(s2) ) == 0)

/*
 * Range Checking and simple math
 */

#define IN_RANGE(xx,ll,rr)      ((xx)>=(ll)&&(xx)<=(rr))
#define CLAMP(xx,ll,rr)         ((xx) = (xx)<(ll)? (ll): (xx)>(rr)? (rr): (xx))
#define SWAP(aa,bb,tt)  ( tt = aa, aa = bb, bb = tt )

#define MAX(aa,bb)      ((aa)>=(bb)?(aa):(bb))
#define MIN(aa,bb)      ((aa)<=(bb)?(aa):(bb))


#define ABS(aa)                 ((aa)>0?(aa):-(aa))
#define SQR(aa)                 ((aa)*(aa))
#define SIGN(aa)                ((aa)>=0? 1: -1)


/*
 * boolean values
 */

#ifndef OK
#define OK      0
#endif
#ifndef ERR
#define ERR    -1
#endif
#ifndef FALSE
#define FALSE   0
#endif
#ifndef TRUE
#define TRUE    1
#endif
#ifndef ON
#define ON      1
#endif
#ifndef OFF
#define OFF     0
#endif


#define __U32__
typedef unsigned       u32;
typedef unsigned short u16;
typedef unsigned char  u8;

#if defined __cplusplus
extern "C" {
#endif

extern int acq200_debug;

#if defined __cplusplus
};
#endif


#ifndef FN
#define FN __FUNCTION__
#endif

#include <sys/syslog.h>

	
#ifndef PROCLOGNAME
#define PROCLOGNAME "acq200control"
#endif
	
#define ACQ200_SYSLOG_SCREEN 1
#define ACQ200_SYSLOG_SYSLOG 2

#ifndef ACQ200_SYSLOG_MODE
#define ACQ200_SYSLOG_MODE (ACQ200_SYSLOG_SCREEN|ACQ200_SYSLOG_SYSLOG)
#endif

#if ((ACQ200_SYSLOG_MODE & ACQ200_SYSLOG_SCREEN) != 0)
#define _ACQ200_SYSLOG_SCREEN(pri, fmt, args...) fprintf(stderr, fmt, ##args)
#else
#define _ACQ200_SYSLOG_SCREEN(pri, fmt, args...)
#endif

#if ((ACQ200_SYSLOG_MODE & ACQ200_SYSLOG_SYSLOG) != 0)
#define _ACQ200_SYSLOG_SYSLOG(pri, fmt, args...)\
do {						\
	openlog(PROCLOGNAME, 0, LOG_USER);	\
	syslog(pri, fmt, ## args);		\
	closelog();				\
} while (0)					
#else
#define _ACQ200_SYSLOG_SYSLOG(pri, fmt, args...)
#endif


#define ACQ200_SYSLOG(pri, fmt, args...)			\
	do {							\
		_ACQ200_SYSLOG_SCREEN(pri, fmt, ## args);	\
		_ACQ200_SYSLOG_SYSLOG(pri, fmt, ## args);	\
	}while(0)

//#define info(fmt, arg...) printf("%s:" fmt "\n", FN, ## arg)
#define info(format, arg...)					\
        ACQ200_SYSLOG(LOG_INFO, "%s " format "\n", FN, ## arg )

#define err(format, arg...) \
        ACQ200_SYSLOG(LOG_ERR, "%s ERROR:" format "\n", FN, ## arg )

#define dbg(lvl, format, arg...)					\
        do {								\
                if(acq200_debug>=lvl ){					\
			if (lvl <= 1){					\
			        ACQ200_SYSLOG( LOG_DEBUG,		\
                                               "%s " format "\n",	\
				               FN, ## arg );		\
			}else{						\
				fprintf(stderr,				\
					"deb(%d) %s " format "\n",	\
					lvl, FN, ## arg );		\
			}						\
                }							\
        } while(0)

static inline char* chomp(char *src) {
	char *plast = src + strlen(src) - 1;
	while(*plast == '\n' || *plast == '\r'){
		*plast-- = '\0';
		if (plast == src){
			break;
		}
	}
	return src;
}

static inline char* tr(char *src, char c1, char c2){
	int ii = 0;
	int i2 = strlen(src);
	for (ii = 0; ii < i2; ii++){
		if (src[ii] == c1){
			src[ii] = c2;
		}
	}
	return src;
}

#define dbgnl(lvl, format, arg...)                                           \
        do {                                                                 \
                if(acq200_debug>=lvl ){                                      \
                        if (lvl <= 1)                                        \
                                ACQ200_SYSLOG( LOG_DEBUG, "%s " format,      \
                                               FN, ## arg );                 \
                        fprintf(stderr,                                      \
                                "deb(%d) %s " format,                        \
                                lvl, FN, ## arg );                           \
                }                                                            \
        } while(0)





#endif
