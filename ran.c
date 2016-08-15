

/*Library of random number generating functions.

VERSION 9-25-99


Includes:
Randomize()              (seed random number generator)
RandomReal(low,high)     (uniform)
RandomInteger(low,high)  (uniform integer in [low,high])
rnd()                    (uniform in (0,1))
RGamma(n,lambda)         (gamma)
RDirichlet(a[],k,b[])    (dirichlet)
RPoisson(mean)           (poisson)
RNormal(mean,sd)        (Normal)
RExpon(mean)             (Exponential)
snorm()                  (Standard normal)
Binomial(n, p)           (Binomial rv)


extern void Randomize(void);  
extern double RandomReal(double low, double high);
extern int RandomInteger(int low, int high);
extern double rnd();
extern double RGamma(double n,double lambda);
extern void RDirichlet(const double * a, const int k, double * b);
extern long RPoisson(double mu);
extern double RExpon(double av);
extern double RNormal(double mu,double sd) ;
extern double fsign( double num, double sign );
extern double sexpo(void);
extern double snorm();
extern double genexp(double av);   
extern long ignpoi(double mean);  
extern long ignuin(int low, int high);   
extern double genunf(double low, double high);   
extern long Binomial(int n, double p)

MORE DETAILS BELOW:





 Random number functions from random.c by Eric Roberts 
void Randomize(void);    
                 Seed the random number generator 
double RandomReal(double low, double high);
                 Get a random number between low and high 
int RandomInteger(int low, int high);
                 Get a random integer between low and high INCLUSIVE
double rnd();
                 Uniform(0,1) random number generation


 Random number functions from Matthew Stephens 
  
double RGamma(double n,double lambda);
                gamma random generator from Ripley, 1987, P230 
void RDirichlet(const double * a, const int k, double * b);
                Dirichlet random generator
   a and b are arrays of length k, containing doubles.
   a is the array of parameters
   b is the output array, where b ~ Dirichlet(a)  


Functions from Brown and Lovato

long RPoisson(double mu);
                 Poisson with parameter mu
double RExpon(double av);
                 exponential with parameter av
double RNormal(double mu,double sigsq) ;
                 Normal with mean mu, var sigsq; by JKP
 ---------Helper functions from Brown and Lovato
double fsign( double num, double sign );
double sexpo(void);
                 exponential with parameter 1
double snorm(); 
                 standard normal N(0,1)  


double genexp(double av);   return RExpon(av); 
long ignpoi(double mean);   return RPoisson(mean); 
long ignuin(int low, int high);    return RandomInteger(low,high);
double genunf(double low, double high);   return RandomReal(low,high);
*/



#include "ran.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ABS(x) ((x) >= 0 ? (x) : -(x))
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))
#define OVERFLO 1e100
#define UNDERFLO 1e-100
/*#define OVERFLOW   1e100 */ /*take log of likelihood when product is more than this*/
/*#define UNDERFLOW  1e-100 */

#include <stdint.h>

/* This is the successor to xorshift128+. It is the fastest full-period
generator passing BigCrush without systematic failures, but due to the
relatively short period it is acceptable only for applications with a
mild amount of parallelism; otherwise, use a xorshift1024* generator.

Beside passing BigCrush, this generator passes the PractRand test suite
up to (and included) 16TB, with the exception of binary rank tests,
which fail due to the lowest bit being an LFSR; all other bits pass all
tests. We suggest to use a sign test to extract a random Boolean value.

Note that the generator uses a simulated rotate operation, which most C
compilers will turn into a single instruction. In Java, you can use
Long.rotateLeft(). In languages that do not make low-level rotation
instructions accessible xorshift128+ could be faster.

The state must be seeded so that it is not everywhere zero. If you have
a 64-bit seed, we suggest to seed a splitmix64 generator and use its
output to fill s. */

static uint64_t s[2];

static inline uint64_t rotl(const uint64_t x, int k) {
	return (x << k) | (x >> (64 - k));
}

static inline uint64_t next(void) {
	const uint64_t s0 = s[0];
	uint64_t s1 = s[1];
	const uint64_t result = s0 + s1;

	s1 ^= s0;
	s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14); // a, b
	s[1] = rotl(s1, 36); // c

	return result;
}

static inline double double_oo(uint64_t x) {
	union { uint64_t i; double d; } u;
	u.i = UINT64_C(0x3ff0000000000001) | x >> 12 ;
	return u.d - 1.0;
}

static inline double double_co(uint64_t x) {
	union { uint64_t i; double d; } u;
	u.i = UINT64_C(0x3ff0000000000000) | x >> 12;
	return u.d - 1.0;
}


/// from TinyMT
/**
* This function represents a function used in the initialization
* by init_by_array
* @param[in] x 64-bit integer
* @return 64-bit integer
*/
static uint64_t ini_func1(uint64_t x) {
	return (x ^ (x >> 59)) * UINT64_C(2173292883993);
}

/**
* This function represents a function used in the initialization
* by init_by_array
* @param[in] x 64-bit integer
* @return 64-bit integer
*/
static uint64_t ini_func2(uint64_t x) {
	return (x ^ (x >> 59)) * UINT64_C(58885565329898161);
}

/**
* This function outputs floating point number from internal state.
* Users should not call this function directly.
* @param random tinymt internal status
* @return floating point number r (1.0 < r < 2.0)
*/
/*
#define TINYMT64_SH8 8
inline static double tinymt64_temper_conv_open() {
	uint64_t x;
	union {
		uint64_t u;
		double d;
	} conv;
	x = s[0] ^ s[1];
	x ^= s[0] >> TINYMT64_SH8;
	conv.u = ((x ^ (-((int64_t)(x & 1)) & random->tmat)) >> 12)
		| UINT64_C(0x3ff0000000000001);
	return conv.d;
}
*/

/*==============================================*/
/*==============================================*/
/*==============================================*/

/* Random number functions (from random.c by Eric Roberts) */

/*Melissa modified in 1/08 so that it either calls srand with given seed or generates one*/
void Randomize(int RANDOMIZE, int *seed)
/* Seed the random number generator */
{   
  FILE *outfile;
  if (RANDOMIZE) {
	  *seed = (int)time(NULL);
  }
  s[0] = ini_func1(*seed);
  s[1] = ini_func2(*seed);

  outfile = fopen("seed.txt", "a");
  fprintf(outfile, "%i\n", *seed);
  fclose(outfile);
}
/*-------------------------------------*/

/*=======================================================*/
/*  Uniform(0,1) random number generation*/

inline double rnd()
{
	return double_oo(next());
}

double RandomReal(double low, double high)
/* Get a random number between low and high */

{
  return (low + rnd() * (high - low) );
}

/*-------------------------------------*/
int RandomInteger(int low, int high)
/* Get a random integer between low and high INCLUSIVE*/
{
  int k;
  k = (int) (rnd() * (high - low + 1));
  return (low + k);
}

/*-----------radnom gamma variate from GSL----------*/

/* position of right-most step */
#define PARAM_R 3.44428647676

/* tabulated values for the heigt of the Ziggurat levels */
static const double ytab[128] = {
	1, 0.963598623011, 0.936280813353, 0.913041104253,
	0.892278506696, 0.873239356919, 0.855496407634, 0.838778928349,
	0.822902083699, 0.807732738234, 0.793171045519, 0.779139726505,
	0.765577436082, 0.752434456248, 0.739669787677, 0.727249120285,
	0.715143377413, 0.703327646455, 0.691780377035, 0.68048276891,
	0.669418297233, 0.65857233912, 0.647931876189, 0.637485254896,
	0.62722199145, 0.617132611532, 0.607208517467, 0.597441877296,
	0.587825531465, 0.578352913803, 0.569017984198, 0.559815170911,
	0.550739320877, 0.541785656682, 0.532949739145, 0.524227434628,
	0.515614886373, 0.507108489253, 0.498704867478, 0.490400854812,
	0.482193476986, 0.47407993601, 0.466057596125, 0.458123971214,
	0.450276713467, 0.442513603171, 0.434832539473, 0.427231532022,
	0.419708693379, 0.41226223212, 0.404890446548, 0.397591718955,
	0.390364510382, 0.383207355816, 0.376118859788, 0.369097692334,
	0.362142585282, 0.355252328834, 0.348425768415, 0.341661801776,
	0.334959376311, 0.328317486588, 0.321735172063, 0.31521151497,
	0.308745638367, 0.302336704338, 0.29598391232, 0.289686497571,
	0.283443729739, 0.27725491156, 0.271119377649, 0.265036493387,
	0.259005653912, 0.253026283183, 0.247097833139, 0.241219782932,
	0.235391638239, 0.229612930649, 0.223883217122, 0.218202079518,
	0.212569124201, 0.206983981709, 0.201446306496, 0.195955776745,
	0.190512094256, 0.185114984406, 0.179764196185, 0.174459502324,
	0.169200699492, 0.1639876086, 0.158820075195, 0.153697969964,
	0.148621189348, 0.143589656295, 0.138603321143, 0.133662162669,
	0.128766189309, 0.123915440582, 0.119109988745, 0.114349940703,
	0.10963544023, 0.104966670533, 0.100343857232, 0.0957672718266,
	0.0912372357329, 0.0867541250127, 0.082318375932, 0.0779304915295,
	0.0735910494266, 0.0693007111742, 0.065060233529, 0.0608704821745,
	0.056732448584, 0.05264727098, 0.0486162607163, 0.0446409359769,
	0.0407230655415, 0.0368647267386, 0.0330683839378, 0.0293369977411,
	0.0256741818288, 0.0220844372634, 0.0185735200577, 0.0151490552854,
	0.0118216532614, 0.00860719483079, 0.00553245272614, 0.00265435214565
};

/* tabulated values for 2^24 times x[i]/x[i+1],
* used to accept for U*x[i+1]<=x[i] without any floating point operations */
static const unsigned long ktab[128] = {
	0, 12590644, 14272653, 14988939,
	15384584, 15635009, 15807561, 15933577,
	16029594, 16105155, 16166147, 16216399,
	16258508, 16294295, 16325078, 16351831,
	16375291, 16396026, 16414479, 16431002,
	16445880, 16459343, 16471578, 16482744,
	16492970, 16502368, 16511031, 16519039,
	16526459, 16533352, 16539769, 16545755,
	16551348, 16556584, 16561493, 16566101,
	16570433, 16574511, 16578353, 16581977,
	16585398, 16588629, 16591685, 16594575,
	16597311, 16599901, 16602354, 16604679,
	16606881, 16608968, 16610945, 16612818,
	16614592, 16616272, 16617861, 16619363,
	16620782, 16622121, 16623383, 16624570,
	16625685, 16626730, 16627708, 16628619,
	16629465, 16630248, 16630969, 16631628,
	16632228, 16632768, 16633248, 16633671,
	16634034, 16634340, 16634586, 16634774,
	16634903, 16634972, 16634980, 16634926,
	16634810, 16634628, 16634381, 16634066,
	16633680, 16633222, 16632688, 16632075,
	16631380, 16630598, 16629726, 16628757,
	16627686, 16626507, 16625212, 16623794,
	16622243, 16620548, 16618698, 16616679,
	16614476, 16612071, 16609444, 16606571,
	16603425, 16599973, 16596178, 16591995,
	16587369, 16582237, 16576520, 16570120,
	16562917, 16554758, 16545450, 16534739,
	16522287, 16507638, 16490152, 16468907,
	16442518, 16408804, 16364095, 16301683,
	16207738, 16047994, 15704248, 15472926
};

/* tabulated values of 2^{-24}*x[i] */
static const double wtab[128] = {
	1.62318314817e-08, 2.16291505214e-08, 2.54246305087e-08, 2.84579525938e-08,
	3.10340022482e-08, 3.33011726243e-08, 3.53439060345e-08, 3.72152672658e-08,
	3.8950989572e-08, 4.05763964764e-08, 4.21101548915e-08, 4.35664624904e-08,
	4.49563968336e-08, 4.62887864029e-08, 4.75707945735e-08, 4.88083237257e-08,
	5.00063025384e-08, 5.11688950428e-08, 5.22996558616e-08, 5.34016475624e-08,
	5.44775307871e-08, 5.55296344581e-08, 5.65600111659e-08, 5.75704813695e-08,
	5.85626690412e-08, 5.95380306862e-08, 6.04978791776e-08, 6.14434034901e-08,
	6.23756851626e-08, 6.32957121259e-08, 6.42043903937e-08, 6.51025540077e-08,
	6.59909735447e-08, 6.68703634341e-08, 6.77413882848e-08, 6.8604668381e-08,
	6.94607844804e-08, 7.03102820203e-08, 7.11536748229e-08, 7.1991448372e-08,
	7.2824062723e-08, 7.36519550992e-08, 7.44755422158e-08, 7.52952223703e-08,
	7.61113773308e-08, 7.69243740467e-08, 7.77345662086e-08, 7.85422956743e-08,
	7.93478937793e-08, 8.01516825471e-08, 8.09539758128e-08, 8.17550802699e-08,
	8.25552964535e-08, 8.33549196661e-08, 8.41542408569e-08, 8.49535474601e-08,
	8.57531242006e-08, 8.65532538723e-08, 8.73542180955e-08, 8.8156298059e-08,
	8.89597752521e-08, 8.97649321908e-08, 9.05720531451e-08, 9.138142487e-08,
	9.21933373471e-08, 9.30080845407e-08, 9.38259651738e-08, 9.46472835298e-08,
	9.54723502847e-08, 9.63014833769e-08, 9.71350089201e-08, 9.79732621669e-08,
	9.88165885297e-08, 9.96653446693e-08, 1.00519899658e-07, 1.0138063623e-07,
	1.02247952126e-07, 1.03122261554e-07, 1.04003996769e-07, 1.04893609795e-07,
	1.05791574313e-07, 1.06698387725e-07, 1.07614573423e-07, 1.08540683296e-07,
	1.09477300508e-07, 1.1042504257e-07, 1.11384564771e-07, 1.12356564007e-07,
	1.13341783071e-07, 1.14341015475e-07, 1.15355110887e-07, 1.16384981291e-07,
	1.17431607977e-07, 1.18496049514e-07, 1.19579450872e-07, 1.20683053909e-07,
	1.21808209468e-07, 1.2295639141e-07, 1.24129212952e-07, 1.25328445797e-07,
	1.26556042658e-07, 1.27814163916e-07, 1.29105209375e-07, 1.30431856341e-07,
	1.31797105598e-07, 1.3320433736e-07, 1.34657379914e-07, 1.36160594606e-07,
	1.37718982103e-07, 1.39338316679e-07, 1.41025317971e-07, 1.42787873535e-07,
	1.44635331499e-07, 1.4657889173e-07, 1.48632138436e-07, 1.50811780719e-07,
	1.53138707402e-07, 1.55639532047e-07, 1.58348931426e-07, 1.61313325908e-07,
	1.64596952856e-07, 1.68292495203e-07, 1.72541128694e-07, 1.77574279496e-07,
	1.83813550477e-07, 1.92166040885e-07, 2.05295471952e-07, 2.22600839893e-07
};


double
gsl_ran_gaussian_ziggurat(const double sigma)
{
	unsigned long int i, j;
	int sign;
	double x, y;

	while (1)
	{
		unsigned long int k = next();
		i = (k & 0xFF);
		j = (k >> 8) & 0xFFFFFF;

		sign = (i & 0x80) ? +1 : -1;
		i &= 0x7f;

		x = j * wtab[i];

		if (j < ktab[i])
			break;

		if (i < 127)
		{
			double y0, y1, U1;
			y0 = ytab[i];
			y1 = ytab[i + 1];
			U1 = double_co(next());
			y = y1 + (y0 - y1) * U1;
		}
		else
		{
			double U1, U2;
			U1 = 1.0 - double_co(next());
			U2 = double_co(next());
			x = PARAM_R - log(U1) / PARAM_R;
			y = exp(-PARAM_R * (x - 0.5 * PARAM_R)) * U2;
		}

		if (y < exp(-0.5 * x * x))
			break;
	}

	return sign * sigma * x;
}

double
RGamma(double a, double b)
{
	/* assume a > 0 */

	if (a < 1)
	{
		double u = rnd();
		return RGamma(1.0 + a, b) * pow(u, 1.0 / a);
	}

	{
		double x, v, u;
		double d = a - 1.0 / 3.0;
		double c = (1.0 / 3.0) / sqrt(d);

		while (1)
		{
			do
			{
				x = gsl_ran_gaussian_ziggurat(1.0);
				v = 1.0 + c * x;
			} while (v <= 0);

			v = v * v * v;
			u = rnd();

			if (u < 1 - 0.0331 * x * x * x * x)
				break;

			if (log(u) < 0.5 * x * x + d * (1 - v + log(v)))
				break;
		}

		return b * d * v;
	}
}

/*-----------Gamma and dirichlet from Matt.----------*/
  /* gamma random generator from Ripley, 1987, P230 */


double RGamma0(double n,double lambda)
{
  double aa;
  double w;
  /*  int i; */

	double x=0.0;
	if(n<1)
	{
		const double E=2.71828182;
		const double b=(n+E)/E;
		double p=0.0;
		one: 
		p=b*rnd();
		if(p>1) goto two;
		x=exp(log(p)/n);
		if(x>-log(rnd())) goto one;
		goto three;
		two: 
		x=-log((b-p)/n);
		if (((n-1)*log(x))<log(rnd())) goto one;
		three:;	
	}
	else if(n==1.0)

	  /* exponential random variable, from Ripley, 1987, P230  */	
	{
		double a=0.0;
		double u,u0,ustar;
	ten:
		u=rnd();
		u0=u;
	twenty:
		ustar=rnd();
		if(u<ustar) goto thirty;
		u=rnd();
		if(u<ustar) goto twenty;
		a++;
		goto ten;
	thirty:
		return (a+u0)/lambda;
	}
	else
	{
		double static nprev=0.0;
		double static c1=0.0;
		double static c2=0.0;
		double static c3=0.0;
		double static c4=0.0;
		double static c5=0.0;
		double u1;
		double u2;
		if(n!=nprev)
		{
			c1=n-1.0;
			aa=1.0/c1;
			c2=aa*(n-1/(6*n));
			c3=2*aa;
			c4=c3+2;
			if(n>2.5) c5=1/sqrt(n);
		}
		four:
		u1=rnd();
		u2=rnd();
		if(n<=2.5) goto five;
		u1=u2+c5*(1-1.86*u1);
		if ((u1<=0) || (u1>=1)) goto four;
		five:
		w=c2*u2/u1;
		if(c3*u1+w+1.0/w < c4) goto six;
		if(c3*log(u1)-log(w)+w >=1) goto four;
		six:
		x=c1*w;		
		nprev=n;
	}	

	return x/lambda;
}


/*
double
LogRGamma (double n, double lambda)
{
  //double aa;
  //  double w;
  //  int i;
  double logx;
  //  return log(RGamma(n, lambda));
  if (n < 1)
  //this is the case we need to worry about underflow
  //copied code from down below but work with logx
  //instead of x
    {
      const double E = 2.71828182;
      const double b = (n + E) / E;
      double p = 0.0;
    one:
      p = b * rnd ();
      if (p > 1)
        goto two;
      logx =  log (p) / n;
      if (logx > log(-log (rnd ())))
        goto one;
      goto three;
    two:
      logx = log(-log (b - p)) -log(n);

      if (((n - 1) * logx) < log (rnd ()))
        goto one;
    three:
return logx-log(lambda);
}
else
//otherwise log the standard version 
return log(RGamma(n,lambda));
}*/




/* Melissa's version, adapted from an algorithm on wikipedia.  January 08 */
double LogRGamma(double n, double lambda)
{
  double v0, v[3], E=2.71828182, em, logem, lognm;
  int i;
  if (lambda!=1.0) {
    printf("lambda=%e!\n", lambda); exit(-1);
  }
  if (n >= 1.0) {
    return log(RGamma(n, lambda));
  }
  v0 = E/(E+n);
  while (1) {
    for (i=0; i<3; i++) {
      v[i] = rnd();
    }
    
    if (v[0] <= v0) {
      logem = 1.0/n*log(v[1]);
      em = exp(logem);
      lognm = log(v[2])+(n-1)*logem;
    } else {
      em = 1.0-log(v[1]);
      logem = log(em);
      lognm = log(v[2]) - em;
    }
    if (lognm <= (n-1)*logem - em) {
      return logem - log(lambda);
    }
  }
}



/*--------------------------------------*/

/* Dirichlet random generator
   a and b are arrays of length k, containing doubles.
   a is the array of parameters
   b is the output array, where b ~ Dirichlet(a)  
   */

void RDirichlet(const double * a, const int k, double * b)
{
int i;
	double sum=0.0;
	for(i=0;i<k;i++)
	{
		b[i]=RGamma(a[i],1);
		sum += b[i];
	}
	for(i=0;i<k;i++)
	{
		b[i] /= sum;
	}
}


/*This function returns both a logged and unlogged version
of the dirichlet function. Designed to cope with
underflows in the RGamma function.
made by Daniel
b is the output array and c is a logged version of b*/

void
LogRDirichlet (const double *a, const int k, double *b,double *c)
{
  int i;
  double sum = 0.0;
  double sum2;
  for (i = 0; i < k; i++) {
    c[i] = LogRGamma (a[i], 1);
    b[i]=exp(c[i]);
    sum += b[i];
  }
  
  /* patch added May 2007 to set gene frequencies equal if all draws from the Gamma distribution are very low. Ensures that P and logP remain defined in this rare event */
  if(sum<UNDERFLO) {
    for(i=0;i<k;i++) {
      b[i] = 1.0/(double)(k);
      c[i] = log(b[i]);
    }
  } else {
    sum2=log(sum);
    for (i = 0; i < k; i++) {
      c[i]-=sum2;
      b[i]/=sum;
    }
  }
}


/*---------------------------------------*/

long RPoisson(double mu)
/*
**********************************************************************
     long RPoissondouble mu)
                    GENerate POIsson random deviate
                              Function
     Generates a single random deviate from a Poisson
     distribution with mean AV.
                              Arguments
     av --> The mean of the Poisson distribution from which
            a random deviate is to be generated.
     RExpon <-- The random deviate.
                              Method
     Renames KPOIS from TOMS as slightly modified by BWB to use RANF
     instead of SUNIF.

     ----substituted rnd for ranf--JKP, 11/98------

     For details see:
               Ahrens, J.H. and Dieter, U.
               Computer Generation of Poisson Deviates
               From Modified Normal Distributions.
               ACM Trans. Math. Software, 8, 2
               (June 1982),163-179
**********************************************************************
**********************************************************************
                                                                      
                                                                      
     P O I S S O N  DISTRIBUTION                                      
                                                                      
                                                                      
**********************************************************************
**********************************************************************
                                                                      
     FOR DETAILS SEE:                                                 
                                                                      
               AHRENS, J.H. AND DIETER, U.                            
               COMPUTER GENERATION OF POISSON DEVIATES                
               FROM MODIFIED NORMAL DISTRIBUTIONS.                    
               ACM TRANS. MATH. SOFTWARE, 8,2 (JUNE 1982), 163 - 179. 
                                                                      
     (SLIGHTLY MODIFIED VERSION OF THE PROGRAM IN THE ABOVE ARTICLE)  
                                                                      
**********************************************************************
      INTEGER FUNCTION RPOISSONIR,MU)
     INPUT:  IR=CURRENT STATE OF BASIC RANDOM NUMBER GENERATOR
             MU=MEAN MU OF THE POISSON DISTRIBUTION
     OUTPUT: IGNPOI=SAMPLE FROM THE POISSON-(MU)-DISTRIBUTION
     MUPREV=PREVIOUS MU, MUOLD=MU AT LAST EXECUTION OF STEP P OR B.
     TABLES: COEFFICIENTS A0-A7 FOR STEP F. FACTORIALS FACT
     COEFFICIENTS A(K) - FOR PX = FK*V*V*SUM(A(K)*V**K)-DEL
     SEPARATION OF CASES A AND B
*/
{
extern double fsign( double num, double sign );
static double a0 = -0.5;
static double a1 = 0.3333333;
static double a2 = -0.2500068;
static double a3 = 0.2000118;
static double a4 = -0.1661269;
static double a5 = 0.1421878;
static double a6 = -0.1384794;
static double a7 = 0.125006;
static double muold = 0.0;
static double muprev = 0.0;
static double fact[10] = {
    1.0,1.0,2.0,6.0,24.0,120.0,720.0,5040.0,40320.0,362880.0
};
static long ignpoi,j,k,kflag,l,m;
static double b1,b2,c,c0,c1,c2,c3,d,del,difmuk,e,fk,fx,fy,g,omega,p,p0,px,py,q,s,
    t,u,v,x,xx,pp[35];

    if(mu == muprev) goto S10;
    if(mu < 10.0) goto S120;
/*
     C A S E  A. (RECALCULATION OF S,D,L IF MU HAS CHANGED)
*/
    muprev = mu;
    s = sqrt(mu);
    d = 6.0*mu*mu;
/*
             THE POISSON PROBABILITIES PK EXCEED THE DISCRETE NORMAL
             PROBABILITIES FK WHENEVER K >= M(MU). L=IFIX(MU-1.1484)
             IS AN UPPER BOUND TO M(MU) FOR ALL MU >= 10 .
*/
    l = (long) (mu-1.1484);
S10:
/*
     STEP N. NORMAL SAMPLE - SNORM(IR) FOR STANDARD NORMAL DEVIATE
*/
    g = mu+s*snorm();
    if(g < 0.0) goto S20;
    ignpoi = (long) (g);
/*
     STEP I. IMMEDIATE ACCEPTANCE IF IGNPOI IS LARGE ENOUGH
*/
    if(ignpoi >= l) return ignpoi;
/*
     STEP S. SQUEEZE ACCEPTANCE - Srnd(IR) FOR (0,1)-SAMPLE U
*/
    fk = (double)ignpoi;
    difmuk = mu-fk;
    u = rnd();  /*was ranf -- JKP*/
    if(d*u >= difmuk*difmuk*difmuk) return ignpoi;
S20:
/*
     STEP P. PREPARATIONS FOR STEPS Q AND H.
             (RECALCULATIONS OF PARAMETERS IF NECESSARY)
             .3989423=(2*PI)**(-.5)  .416667E-1=1./24.  .1428571=1./7.
             THE QUANTITIES B1, B2, C3, C2, C1, C0 ARE FOR THE HERMITE
             APPROXIMATIONS TO THE DISCRETE NORMAL PROBABILITIES FK.
             C=.1069/MU GUARANTEES MAJORIZATION BY THE 'HAT'-FUNCTION.
*/
    if(mu == muold) goto S30;
    muold = mu;
    omega = 0.3989423/s;
    b1 = 4.166667E-2/mu;
    b2 = 0.3*b1*b1;
    c3 = 0.1428571*b1*b2;
    c2 = b2-15.0*c3;
    c1 = b1-6.0*b2+45.0*c3;
    c0 = 1.0-b1+3.0*b2-15.0*c3;
    c = 0.1069/mu;
S30:
    if(g < 0.0) goto S50;
/*
             'SUBROUTINE' F IS CALLED (KFLAG=0 FOR CORRECT RETURN)
*/
    kflag = 0;
    goto S70;
S40:
/*
     STEP Q. QUOTIENT ACCEPTANCE (RARE CASE)
*/
    if(fy-u*fy <= py*exp(px-fx)) return ignpoi;
S50:
/*
     STEP E. EXPONENTIAL SAMPLE - SEXPO(IR) FOR STANDARD EXPONENTIAL
             DEVIATE E AND SAMPLE T FROM THE LAPLACE 'HAT'
             (IF T <= -.6744 THEN PK < FK FOR ALL MU >= 10.)
*/
    e = sexpo();
    u = rnd();  /*was ranf--JKP*/
    u += (u-1.0);
    t = 1.8+fsign(e,u);
    if(t <= -0.6744) goto S50;
    ignpoi = (long) (mu+s*t);
    fk = (double)ignpoi;
    difmuk = mu-fk;
/*
             'SUBROUTINE' F IS CALLED (KFLAG=1 FOR CORRECT RETURN)
*/
    kflag = 1;
    goto S70;
S60:
/*
     STEP H. HAT ACCEPTANCE (E IS REPEATED ON REJECTION)
*/
    if(c*fabs(u) > py*exp(px+e)-fy*exp(fx+e)) goto S50;
    return ignpoi;
S70:
/*
     STEP F. 'SUBROUTINE' F. CALCULATION OF PX,PY,FX,FY.
             CASE IGNPOI .LT. 10 USES FACTORIALS FROM TABLE FACT
*/
    if(ignpoi >= 10) goto S80;
    px = -mu;
    py = pow(mu,(double)ignpoi)/ *(fact+ignpoi);
    goto S110;
S80:
/*
             CASE IGNPOI .GE. 10 USES POLYNOMIAL APPROXIMATION
             A0-A7 FOR ACCURACY WHEN ADVISABLE
             .8333333E-1=1./12.  .3989423=(2*PI)**(-.5)
*/
    del = 8.333333E-2/fk;
    del -= (4.8*del*del*del);
    v = difmuk/fk;
    if(fabs(v) <= 0.25) goto S90;
    px = fk*log(1.0+v)-difmuk-del;
    goto S100;
S90:
    px = fk*v*v*(((((((a7*v+a6)*v+a5)*v+a4)*v+a3)*v+a2)*v+a1)*v+a0)-del;
S100:
    py = 0.3989423/sqrt(fk);
S110:
    x = (0.5-difmuk)/s;
    xx = x*x;
    fx = -0.5*xx;
    fy = omega*(((c3*xx+c2)*xx+c1)*xx+c0);
    if(kflag <= 0) goto S40;
    goto S60;
S120:
/*
     C A S E  B. (START NEW TABLE AND CALCULATE P0 IF NECESSARY)
*/
    muprev = 0.0;
    if(mu == muold) goto S130;
    muold = mu;
    m = max(1L,(long) (mu));
    l = 0;
    p = exp(-mu);
    q = p0 = p;
S130:
/*
     STEP U. UNIFORM SAMPLE FOR INVERSION METHOD
*/
    u = rnd();  /*was ranf here-- JKP*/
    ignpoi = 0;
    if(u <= p0) return ignpoi;
/*
     STEP T. TABLE COMPARISON UNTIL THE END PP(L) OF THE
             PP-TABLE OF CUMULATIVE POISSON PROBABILITIES
             (0.458=PP(9) FOR MU=10)
*/
    if(l == 0) goto S150;
    j = 1;
    if(u > 0.458) j = min(l,m);
    for(k=j; k<=l; k++) {
        if(u <= *(pp+k-1)) goto S180;
    }
    if(l == 35) goto S130;
S150:
/*
     STEP C. CREATION OF NEW POISSON PROBABILITIES P
             AND THEIR CUMULATIVES Q=PP(K)
*/
    l += 1;
    for(k=l; k<=35; k++) {
        p = p*mu/(double)k;
        q += p;
        *(pp+k-1) = q;
        if(u <= q) goto S170;
    }
    l = 35;
    goto S130;
S170:
    l = k;
S180:
    ignpoi = k;
    return ignpoi;
}

/*-----------------------------------*/
double RNormal(double mu,double sd) 
     /* Returns Normal rv with mean mu, variance sigsq.   
        Uses snorm function of Brown and Lovato.  By JKP*/
{

  return (mu + sd*snorm());

}
/*
**********************************************************************
                                                                      
                                                                      
     (STANDARD-)  N O R M A L  DISTRIBUTION                           
                                                                      
                                                                      
**********************************************************************
**********************************************************************
                                                                      
     FOR DETAILS SEE:                                                 
                                                                      
               AHRENS, J.H. AND DIETER, U.                            
               EXTENSIONS OF FORSYTHE'S METHOD FOR RANDOM             
               SAMPLING FROM THE NORMAL DISTRIBUTION.                 
               MATH. COMPUT., 27,124 (OCT. 1973), 927 - 937.          
                                                                      
     ALL STATEMENT NUMBERS CORRESPOND TO THE STEPS OF ALGORITHM 'FL'  
     (M=5) IN THE ABOVE PAPER     (SLIGHTLY MODIFIED IMPLEMENTATION)  
                                                                      
     Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of   
     SUNIF.  The argument IR thus goes away.                          
                                                                      
**********************************************************************
     THE DEFINITIONS OF THE CONSTANTS A(K), D(K), T(K) AND
     H(K) ARE ACCORDING TO THE ABOVEMENTIONED ARTICLE
*/
double snorm()    /*was snorm(void) -- JKP*/
{
static double a[32] = {
    0.0,3.917609E-2,7.841241E-2,0.11777,0.1573107,0.1970991,0.2372021,0.2776904,
    0.3186394,0.36013,0.4022501,0.4450965,0.4887764,0.5334097,0.5791322,
    0.626099,0.6744898,0.7245144,0.7764218,0.8305109,0.8871466,0.9467818,
    1.00999,1.077516,1.150349,1.229859,1.318011,1.417797,1.534121,1.67594,
    1.862732,2.153875
};
static double d[31] = {
    0.0,0.0,0.0,0.0,0.0,0.2636843,0.2425085,0.2255674,0.2116342,0.1999243,
    0.1899108,0.1812252,0.1736014,0.1668419,0.1607967,0.1553497,0.1504094,
    0.1459026,0.14177,0.1379632,0.1344418,0.1311722,0.128126,0.1252791,
    0.1226109,0.1201036,0.1177417,0.1155119,0.1134023,0.1114027,0.1095039
};
static double t[31] = {
    7.673828E-4,2.30687E-3,3.860618E-3,5.438454E-3,7.0507E-3,8.708396E-3,
    1.042357E-2,1.220953E-2,1.408125E-2,1.605579E-2,1.81529E-2,2.039573E-2,
    2.281177E-2,2.543407E-2,2.830296E-2,3.146822E-2,3.499233E-2,3.895483E-2,
    4.345878E-2,4.864035E-2,5.468334E-2,6.184222E-2,7.047983E-2,8.113195E-2,
    9.462444E-2,0.1123001,0.136498,0.1716886,0.2276241,0.330498,0.5847031
};
static double h[31] = {
    3.920617E-2,3.932705E-2,3.951E-2,3.975703E-2,4.007093E-2,4.045533E-2,
    4.091481E-2,4.145507E-2,4.208311E-2,4.280748E-2,4.363863E-2,4.458932E-2,
    4.567523E-2,4.691571E-2,4.833487E-2,4.996298E-2,5.183859E-2,5.401138E-2,
    5.654656E-2,5.95313E-2,6.308489E-2,6.737503E-2,7.264544E-2,7.926471E-2,
    8.781922E-2,9.930398E-2,0.11556,0.1404344,0.1836142,0.2790016,0.7010474
};
static long i;
static double snorm,u,s,ustar,aa,w,y,tt;
    u = rnd();   /* was ranf--JKP*/
    s = 0.0;
    if(u > 0.5) s = 1.0;
    u += (u-s);
    u = 32.0*u;
    i = (long) (u);
    if(i == 32) i = 31;
    if(i == 0) goto S100;
/*
                                START CENTER
*/
    ustar = u-(double)i;
    aa = *(a+i-1);
S40:
    if(ustar <= *(t+i-1)) goto S60;
    w = (ustar-*(t+i-1))**(h+i-1);
S50:
/*
                                EXIT   (BOTH CASES)
*/
    y = aa+w;
    snorm = y;
    if(s == 1.0) snorm = -y;
    return snorm;
S60:
/*
                                CENTER CONTINUED
*/
    u = rnd();                /*was ranf--JKP*/
    w = u*(*(a+i)-aa);
    tt = (0.5*w+aa)*w;
    goto S80;
S70:
    tt = u;
    ustar = rnd();                /*was ranf--JKP*/
S80:
    if(ustar > tt) goto S50;
    u = rnd();               /*was ranf--JKP*/
    if(ustar >= u) goto S70;
    ustar = rnd();               /*was ranf--JKP*/
    goto S40;
S100:
/*
                                START TAIL
*/
    i = 6;
    aa = *(a+31);
    goto S120;
S110:
    aa += *(d+i-1);
    i += 1;
S120:
    u += u;
    if(u < 1.0) goto S110;
    u -= 1.0;
S140:
    w = u**(d+i-1);
    tt = (0.5*w+aa)*w;
    goto S160;
S150:
    tt = u;
S160:
    ustar = rnd();               /*was ranf--JKP*/
    if(ustar > tt) goto S50;
    u = rnd();               /*was ranf--JKP*/
    if(ustar >= u) goto S150;
    u = rnd();               /*was ranf--JKP*/
    goto S140;
}

/*
**********************************************************************
     double RExpon(double av)
                    GENerate EXPonential random deviate
                              Function
     Generates a single random deviate from an exponential
     distribution with mean AV.
                              Arguments
     av --> The mean of the exponential distribution from which
            a random deviate is to be generated.
                              Method
     Renames SEXPO from TOMS as slightly modified by BWB to use RANF
     instead of SUNIF.
     For details see:
               Ahrens, J.H. and Dieter, U.
               Computer Methods for Sampling From the
               Exponential and Normal Distributions.
               Comm. ACM, 15,10 (Oct. 1972), 873 - 882.
**********************************************************************
*/
double RExpon(double av)
{
static double RExpon;

    RExpon = sexpo()*av;
    return RExpon;
}

/*
**********************************************************************
                                                                      
                                                                      
     (STANDARD-)  E X P O N E N T I A L   DISTRIBUTION                
                                                                      
                                                                      
**********************************************************************
**********************************************************************
                                                                      
     FOR DETAILS SEE:                                                 
                                                                      
               AHRENS, J.H. AND DIETER, U.                            
               COMPUTER METHODS FOR SAMPLING FROM THE                 
               EXPONENTIAL AND NORMAL DISTRIBUTIONS.                  
               COMM. ACM, 15,10 (OCT. 1972), 873 - 882.               
                                                                      
     ALL STATEMENT NUMBERS CORRESPOND TO THE STEPS OF ALGORITHM       
     'SA' IN THE ABOVE PAPER (SLIGHTLY MODIFIED IMPLEMENTATION)       
                                                                      
     Modified by Barry W. Brown, Feb 3, 1988 to use RANF instead of   
     SUNIF.  The argument IR thus goes away.                          
                                                                      
**********************************************************************
     Q(N) = SUM(ALOG(2.0)**K/K!)    K=1,..,N ,      THE HIGHEST N
     (HERE 8) IS DETERMINED BY Q(N)=1.0 WITHIN STANDARD PRECISION
*/
double sexpo(void)
{
static double q[8] = {
    0.6931472,0.9333737,0.9888778,0.9984959,0.9998293,0.9999833,0.9999986,1.0
};
static long i;
static double sexpo,a,u,ustar,umin;
static double *q1 = q;
    a = 0.0;
    u = rnd();  /* was ranf--JKP */
    goto S30;
S20:
    a += *q1;
S30:
    u += u;
    if(u <= 1.0) goto S20;
    u -= 1.0;
    if(u > *q1) goto S60;
    sexpo = a+u;
    return sexpo;
S60:
    i = 1;
    ustar = rnd();
    umin = ustar;
S70:
    ustar = rnd();  /* was ranf--JKP */
    if(ustar < umin) umin = ustar;
    i += 1;
    if(u > *(q+i-1)) goto S70;
    sexpo = a+umin**q1;
    return sexpo;
}

/*------------------------------------*/
double fsign( double num, double sign )
/* Transfers sign of argument sign to argument num */
{
if ( ( sign>0.0f && num<0.0f ) || ( sign<0.0f && num>0.0f ) )
    return -num;
else return num;
}

/*------------------------------------*/
double genexp(double av)
{
  return RExpon(av);
}
/*------------------------------------*/
long ignpoi(double mean)
{
  return RPoisson(mean);
}
/*------------------------------------*/
long ignuin(int low, int high)
{
  return RandomInteger(low,high);
}
/*-------------------------------------*/
double genunf(double low, double high)
{
  return RandomReal(low,high);
}
/*-------------------------------------*/
long Binomial(int n, double p)
/*returns a binomial random number, for the number of successes in n trials
  with prob of sucess p.  There's probably a qicker algorithm than this, but I
  can't see how to write the cumulative prob in a simple form*/
{
  int i,sofar;

  sofar = 0;
  for (i=0; i<n; i++)
    if (rnd() < p) sofar++;
  return sofar;
  
}
/*-------------------------------------*/
long Binomial1(int n, double p)
/*returns a binomial random number, for the number of successes in n
trials with prob of sucess p.  There's probably a qicker algorithm
than this, but I can't see how to write the cumulative prob in a
simple form.  This more complicated algorithm, which involves summing
the probabilities appears to be substantially slower than the
simple-minded approach, above.*/
{
  double cum = 0.0;
  int up,down; 
  /*  double upvalue,downvalue; */
  double rv;
  /*  double q = 1 - p; */

  if (p<=0.0) return 0;  /*trivial cases*/
  if (p>=1.0) return 0;
  if (n<1) return 0;
  
  rv = rnd();            /*random number in (0,1)*/
  up = n*p;              /*start at mean and work out, adding probs to the total (cum)*/
  down = up;
  
  do
    {
      if (up <= n)
	{
	  cum += BinoProb(n,p,up);
	  if (rv <= cum) return up;
	  up++;
	}
      down--;
      if (down >= 0)
	{	  
	  cum += BinoProb(n,p,down);
	  if (rv <= cum) return down;
	}
    }
  while ((up <=n ) || (down >= 1));

  return Binomial(n,p);  /*in case of reaching no result...possibly due to underflow(?)*/
}
/*-------------------------------------*/
double BinoProb(int n, double p,int i)
/*returns the prob of i successes in n trials with prob of sucess p.*/
{

  double logsum = 0.0;
  double runningtotal = 1.0;
  int j;

  if (i>(n-i))  /*figure out the n-choose-i part*/
    {
      for (j=2; j <= (n-i); j++)
	{
	  runningtotal /= j;
	  if (runningtotal<UNDERFLO)
	    {
	      logsum += log(runningtotal);
	      runningtotal = 1.0;
	    }
	}
      for (j=i+1; j <= n; j++)
	{
	  runningtotal *= j;
	  if (runningtotal>OVERFLO)
	    {
	      logsum += log(runningtotal);
	      runningtotal = 1.0;
	    }
	}
    }
  else
    {
      for (j=2; j <= i; j++)
	{
	  runningtotal /= j;
	  if (runningtotal<UNDERFLO)
	    {
	      logsum += log(runningtotal);
	      runningtotal = 1.0;
	    }
	}
      for (j=n-i+1; j <= n; j++)
	{
	  runningtotal *= j;
	  if (runningtotal>OVERFLO)
	    {
	      logsum += log(runningtotal);
	      runningtotal = 1.0;
	    }
	}
    }
  logsum += log(runningtotal);
  logsum += i*log(p);
  logsum += (n-i)*log(1-p);
  
  return exp(logsum);
}
