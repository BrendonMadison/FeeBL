
#include "grvCPP.h"

/* Common Block Declarations */


/* Table of constant values */



double GRVPAR::grvul(double x,double q2)
{
  // Builtin functions 

    /* Local variables */
    static double s, s2;
    static double rts;

/* ********************************************************************** 
*/
/* * Leading order up-quark distributions. X is Bjorken-x, and Q2 is    * 
*/
/* * the factorization scale. Recall that alpha_em has been factored    * 
*/
/* * out. The parametrization is supposed to be valid for x > 10**-5,  * 
*/
/* * Q**2 < 10**6 GeV**2.                                               * 
*/
/* ********************************************************************** 
*/
    s    = log(log(q2 * 18.58) / 1.536);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = 1.717;
    bet_ = .641;
    a_   = .5 - s * .176;
    b_   = 15. - rts * 5.687 - s2 * .552;
    ba_  = rts * .046 + .235;
    bb_  = .082 - s * .051 + s2 * .168;
    c_   = s * .459;
    d_   = .354 - s * .061;
    e_   = s * 1.678 + 4.899;
    ep_  = s * 1.389 + 2.046;
    return fl(x,s);
} /* grvul */

double GRVPAR::grvdl(double x,double q2)
{
  // Builtin functions 

    /* Local variables */
    static double s, s2;
    static double rts;

/* ************************************ */
/* * Same as GRVUL, but for d-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 18.58) / 1.536);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = 1.549;
    bet_ = .782;
    a_   = s * .026 + .496;
    b_   = .685 - rts * .58 + s2 * .608;
    ba_  = s * .302 + .233;
    bb_  = s * -.818 + s2 * .198;
    c_   = s * .154 + .114;
    d_   = .405 - s * .195 + s2 * .046;
    e_   = s * 1.226 + 4.807;
    ep_  = s * .664 + 2.166;
    return fl(x,s);
} /* grvdl */

double GRVPAR::grvgl(double x,double q2)
{
  // Builtin functions 

  // Local variables 
    static double s, s2;
    static double rts;

/* ********************************** */
/* * Same as GRVUL, but for gluons. * */
/* ********************************** */
    s    = log(log(q2 * 18.58) / 1.536);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = .676;
    bet_ = 1.089;
    a_   = .462 - rts * .524;
    b_   = 5.451 - s2 * .804;
    ba_  = .535 - rts * .504 + s2 * .288;
    bb_  = .364 - s * .52;
    c_   = s2 * .115 - .323;
    d_   = s * .79 + .233 - s2 * .139;
    e_   = s * 1.968 + .893;
    ep_  = s * .392 + 3.432;
    return fl(x,s);
} /* grvgl */

double GRVPAR::grvsl(double x,double q2)
{
    /* Builtin functions */
  /*   double log(), sqrt(); */

    /* Local variables */
    static double s, s2;
    static double rts;

/* ************************************ */
/* * Same as GRVUL, but for s-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 18.58) / 1.536);
    rts  = sqrt(s);
    s2   = s * s;
    sp_  = 0.;
    alp_ = 1.609;
    bet_ = .962;
    a_   = .47 - s2 * .099;
    b_   = 3.246;
    ba_  = .121 - rts * .068;
    bb_  = s * .074 - .09;
    c_   = s * .034 + .062;
    d_   = s * .226 - s2 * .06;
    e_   = s * 1.707 + 4.288;
    ep_  = s * .656 + 2.122;
    return fh(x,s);
} /* grvsl */

double GRVPAR::grvcl(double x,double q2)
{
    /* Builtin functions */
  /*  double log(); */

    /* Local variables */
    static double s, s2;

/* ************************************ */
/* * Same as GRVUL, but for c-quarks. * */
/* ************************************ */
    s = log(log(q2 * 18.58) / 1.536);
    s2   = s * s;
    sp_  = .888;
    alp_ = .97;
    bet_ = .545;
    a_   = 1.254 - s * .251;
    b_   = 3.932 - s2 * .327;
    ba_  = s * .202 + .658;
    bb_  = -.699;
    c_   = .965;
    d_   = s * .141 - s2 * .027;
    e_   = s * .969 + 4.911;
    ep_  = s * .952 + 2.796;
    return fh(x,s);
} /* grvcl_ */

double GRVPAR::grvbl(double x,double q2)
{
    /* Builtin functions */
  /*    double log(); */

    /* Local variables */
    static double s;

/* ************************************ */
/* * Same as GRVUL, but for b-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 18.58) / 1.536);
    sp_  = 1.351;
    alp_ = 1.016;
    bet_ = .338;
    a_   = 1.961 - s * .37;
    b_   = s * .119 + .923;
    ba_  = s * .207 + .815;
    bb_  = -2.275;
    c_   = 1.48;
    d_   = s * .173 - .223;
    e_   = s * .623 + 5.426;
    ep_  = s * .901 + 3.819;
    return fh(x,s);
} /* grvbl */

double GRVPAR::grvuh(double x,double q2)
{
    /* System generated locals */
    double ret_val;

    /* Builtin functions */
    /*    double log(), sqrt(), atan(); */

    /* Local variables */
    static double s, s2;
    static double bgamma, eq, pi, rts;

/* ********************************************************************** 
*/
/* * Higher order up-quark distributions. X is Bjorken-x, and Q2 is the * 
*/
/* * factorization scale. Recall that alpha_em has been factored out.   * 
*/
/* * The parametrization is supposed to be  valid for x > 10**-5,       * 
*/
/* * Q**2 < 10**6 GeV**2.                                               * 
*/
/* * WARNING: These distributions are in the DIS_gamma scheme; to trans-* 
*/
/* * late them into MSbar, a term proportional to B_gamma must be added.* 
*/
/* *--------------------------------------------------------------------* 
*/
/* * Modification by J. Zunft on 11/06/92: MS_bar scheme                * 
*/
/* ********************************************************************** 
 */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = .583;
    bet_ = .688;
    a_   = .449 - s * .025 - s2 * .071;
    b_   = 5.06 - rts * 1.116;
    ba_  = .103;
    bb_  = s * .422 + .319;
    c_   = s * 4.792 + 1.508 - s2 * 1.963;
    d_   = rts * .222 + 1.075 - s2 * .193;
    e_   = s * 1.131 + 4.147;
    ep_  = s * .874 + 1.661;
    ret_val = fl(x,s);
    pi = atan(1.) * 4.;
    eq = .66666666666666663;
/* Computing 2nd power */
    bgamma = (((float)1. - x * (float)2. + x * x * (float)2.) * log(((
	    float)1. - x) / x) - (float)1. + x * (float)8. * ((float)1. - 
	    x)) * (float)4.;
/* Computing 2nd power */
    ret_val -= 3.0*eq*eq/(8.0*pi) * bgamma;
    return ret_val;
} /* grvuh_ */

double GRVPAR::grvdh(double x,double q2)
{
    /* System generated locals */
    double ret_val, d__1;

    /* Builtin functions */
    /*    double log(), sqrt(), atan(); */

    /* Local variables */
    static double s, s2;
    static double bgamma, eq, pi, rts;

/* ************************************ */
/* * Same as GRVUH, but for d-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = .591;
    bet_ = .698;
    a_   = .442 - s * .132 - s2 * .058;
    b_   = 5.437 - rts * 1.916;
    ba_  = .099;
    bb_  = .311 - s * .059;
    c_   = s * .078 + .8 - s2 * .1;
    d_   = rts * .294 + .862 - s2 * .184;
    e_   = s * 1.352 + 4.202;
    ep_  = s * .99 + 1.841;
    ret_val = fl(x,s);
    pi = atan(1.) * 4.;
    eq = .33333333333333331;
/* Computing 2nd power */
    d__1 = x;
    bgamma = (((float)1. - x * (float)2. + d__1 * d__1 * (float)2.) * log(((
	    float)1. - x) / x) - (float)1. + x * (float)8. * ((float)1. - 
	    x)) * (float)4.;
/* Computing 2nd power */
    d__1 = eq;
    ret_val -= d__1 * d__1 * 3. / (pi * 8.) * bgamma;
    return ret_val;
} /* grvdh */

double GRVPAR::grvgh(double x,double q2)
{
    /* System generated locals */

    /* Builtin functions */
    /*   double log(), sqrt(); */

    /* Local variables */
    static double s, s2;
    static double rts;

/* **************************************************************** */
/* * Same as GRVUH, but for gluons. Note that DIS_gamma and MSbar * */
/* * scheme are identical for gluons (to the given order).        * */
/* **************************************************************** */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = 1.161;
    bet_ = 1.591;
    a_   = .53 - rts * .742 + s2 * .025;
    b_   = 5.662;
    ba_  = .533 - rts * .281 + s2 * .218;
    bb_  = .025 - s * .518 + s2 * .156;
    c_   = s2 * .209 - .282;
    d_   = s * 1.058 + .107 - s2 * .218;
    e_   = s * 2.704;
    ep_  = 3.071 - s * .378;
    return fl(x,s);
} /* grvgh */

double GRVPAR::grvsh(double x,double q2)
{
    /* System generated locals */
    double ret_val, d__1;

    /* Builtin functions */
    /*   double log(), sqrt(), atan(); */

    /* Local variables */
    static double s, s2;
    static double bgamma, eq, pi, rts;

/* ************************************ */
/* * Same as GRVUH, but for s-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    sp_  = 0.;
    alp_ = .635;
    bet_ = .456;
    a_   = 1.77 - rts * .735 - s2 * .079;
    b_   = 3.832;
    ba_  = .084 - s * .023;
    bb_  = .136;
    c_   = 2.119 - s * .942 + s2 * .063;
    d_   = s * .076 + 1.271 - s2 * .19;
    e_   = s * .737 + 4.604;
    ep_  = s * .976 + 1.641;
    ret_val = fh(x,s);
    pi = atan(1.) * 4.;
    eq = .33333333333333331;
/* Computing 2nd power */
    d__1 = x;
    bgamma = (((float)1. - x * (float)2. + d__1 * d__1 * (float)2.) * log(((
	    float)1. - x) / x) - (float)1. + x * (float)8. * ((float)1. - 
	    x)) * (float)4.;
/* Computing 2nd power */
    d__1 = eq;
    ret_val -= d__1 * d__1 * 3. / (pi * 8.) * bgamma;
    return ret_val;
} /* grvsh_ */

double GRVPAR::grvch(double x,double q2)
{
    /* System generated locals */
    double ret_val, d__1;

    /* Builtin functions */
    /*    double log(), atan(); */

    /* Local variables */
    static double s, s2;
    static double bgamma, eq, pi;

/* ************************************ */
/* * Same as GRVUH, but for c-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 16.26) / 1.585);
    s2   = s * s;
    sp_  = .82;
    alp_ = .926;
    bet_ = .152;
    a_   = 1.142 - s * .175;
    b_   = 3.276;
    ba_  = s * .317 + .504;
    bb_  = -.433;
    c_   = 3.334;
    d_   = s * .326 + .398 - s2 * .107;
    e_   = s * .408 + 5.493;
    ep_  = s * 1.277 + 2.426;
    ret_val = fh(x,s);
    pi = atan(1.) * 4.;
    eq = .66666666666666663;
/* Computing 2nd power */
    d__1 = x;
    bgamma = (((float)1. - x * (float)2. + d__1 * d__1 * (float)2.) * log(((
	    float)1. - x) / x) - (float)1. + x * (float)8. * ((float)1. - 
	    x)) * (float)4.;
/* Computing 2nd power */
    d__1 = eq;
    ret_val -= d__1 * d__1 * 3. / (pi * 8.) * bgamma;
    return ret_val;
} /* grvch_ */

double GRVPAR::grvbh(double x,double q2)
{
    /* System generated locals */
    double ret_val, d__1;

    /* Builtin functions */
    /*   double log(), atan(); */

    /* Local variables */
    static double s;
    static double bgamma, eq, pi;

/* ************************************ */
/* * Same as GRVUH, but for b-quarks. * */
/* ************************************ */
    s    = log(log(q2 * 16.26) / 1.585);
    sp_  = 1.297;
    alp_ = .969;
    bet_ = .266;
    a_   = 1.953 - s * .391;
    b_   = 1.657 - s * .161;
    ba_  = s * .034 + 1.076;
    bb_  = -2.015;
    c_   = 1.662;
    d_   = s * .016 + .353;
    e_   = s * .249 + 5.713;
    ep_  = s * .673 + 3.456;
    ret_val = fh(x,s);
    pi = atan(1.) * 4.;
    eq = .33333333333333331;
/* Computing 2nd power */
    d__1 = x;
    bgamma = (((float)1. - x * (float)2. + d__1 * d__1 * (float)2.) * log(((
	    float)1. - x) / x) - (float)1. + x * (float)8. * ((float)1. - 
	    x)) * (float)4.;
/* Computing 2nd power */
    d__1 = eq;
    ret_val -= d__1 * d__1 * 3. / (pi * 8.) * bgamma;
    return ret_val;
} /* grvbh */

double GRVPAR::grvuh0(double x,double q2)
{
    /* Builtin functions */
  /*   double log(), sqrt(); */

    /* Local variables */
    static double s, s2;
    static double rts;

/* ********************************************************************** 
*/
/* * The leading order part of the total HO up-quark distribution. This * 
*/
/* * part is needed since, strictly speaking, only this part should be  * 
*/
/* * multiplied with the HO part of a NLO cross section or Wilson co-   * 
*/
/* * efficient. For the meaning of the variables, see GRVUH.            * 
*/
/* ********************************************************************** 
*/
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = 1.447;
    bet_ = .848;
    a_   = s * .2 + .527 - s2 * .107;
    b_   = 7.106 - rts * .31 - s2 * .786;
    ba_  = s * .533 + .197;
    bb_  = .062 - s * .398 + s2 * .109;
    c_   = s * .755 - s2 * .112;
    d_   = .318 - s * .059;
    e_   = s * 1.708 + 4.225;
    ep_  = s * .866 + 1.752;
    return fl(x,s);
} /* grvuh0_ */

double GRVPAR::grvdh0(double x,double q2)
{
    /* System generated locals */
    double ret_val;

    /* Builtin functions */
    /*    double log(), sqrt(); */

    /* Local variables */
    static double s, s2;
    static double rts;

/* ************************************* */
/* * Same as GRVUH0, but for d-quarks. * */
/* ************************************* */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = 1.424;
    bet_ = .77;
    a_   = rts * .067 + .5 - s2 * .055;
    b_   = .376 - rts * .453 + s2 * .405;
    ba_  = s * .184 + .156;
    bb_  = s * -.528 + s2 * .146;
    c_   = s * .092 + .121;
    d_   = .379 - s * .301 + s2 * .081;
    e_   = s * 1.638 + 4.346;
    ep_  = s * 1.016 + 1.645;
    ret_val = fl(x,s);
    return ret_val;
} /* grvdh0 */

double GRVPAR::grvgh0(double x,double q2)
{
    /* System generated locals */

    /* Builtin functions */
    /*   double log(), sqrt(); */

    /* Local variables */
    static double s, s2;
    static double rts;

/* ***************************************************************** */
/* * Same as GRVUH0, but for gluons. Note that DIS_gamma and MSbar * */
/* * scheme are identical for gluons (to the given order).         * */
/* ***************************************************************** */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    alp_ = .661;
    bet_ = .793;
    a_   = .537 - rts * .6;
    b_   = 6.389 - s2 * .953;
    ba_  = .558 - rts * .383 + s2 * .261;
    bb_  = s * -.305;
    c_   = s2 * .078 - .222;
    d_   = s * .978 + .153 - s2 * .209;
    e_   = s * 1.772 + 1.429;
    ep_  = s * .806 + 3.331;
    return fl(x,s);
} /* grvgh0 */

double GRVPAR::grvsh0(double x,double q2)
{
    /* Builtin functions */
  /*   double log(), sqrt(); */

    /* Local variables */
    static double s, s2;
    static double rts;

/* ************************************* */
/* * Same as GRVUH0, but for s-quarks. * */
/* ************************************* */
    s    = log(log(q2 * 16.26) / 1.585);
    rts  = sqrt(s);
    s2   = s * s;
    sp_  = 0.;
    alp_ = 1.578;
    bet_ = .863;
    a_   = s * .332 + .622 - s2 * .3;
    b_   = 2.469;
    ba_  = .211 - rts * .064 - s2 * .018;
    bb_  = s * .122 - .215;
    c_   = .153;
    d_   = s * .253 - s2 * .081;
    e_   = s * 2.014 + 3.99;
    ep_  = s * .986 + 1.72;
    return fh(x,s);
} /* grvsh0 */

double GRVPAR::grvch0(double x,double q2)
{
    /* Builtin functions */
  /*   double log(); */

    /* Local variables */
    static double s, s2;

/* ************************************* */
/* * Same as GRVUH0, but for c-quarks. * */
/* ************************************* */
    s    = log(log(q2 * 16.26) / 1.585);
    s2   = s * s;
    sp_  = .82;
    alp_ = .929;
    bet_ = .381;
    a_   = 1.228 - s * .231;
    b_   = 3.806 - s2 * .337;
    ba_  = s * .15 + .932;
    bb_  = -.906;
    c_   = 1.133;
    d_   = s * .138 - s2 * .028;
    e_   = s * .628 + 5.588;
    ep_  = s * 1.054 + 2.665;
    return fh(x,s);
} /* grvch0 */

double GRVPAR::grvbh0(double x,double q2)
{
    /* Builtin functions */
  /*    double log(); */

    /* Local variables */
    static double s;

/* ************************************* */
/* * Same as GRVUH0, but for b-quarks. * */
/* ************************************* */
    s    = log(log(q2 * 16.26) / 1.585);
    sp_  = 1.297;
    alp_ = .97;
    bet_ = .207;
    a_   = 1.719 - s * .292;
    b_   = s * .096 + .928;
    ba_  = s * .178 + .845;
    bb_  = -2.31;
    c_   = 1.558;
    d_   = s * .151 - .191;
    e_   = s * .282 + 6.089;
    ep_  = s * 1.062 + 3.379;
    return fh(x,s);
} /* grvbh0 */
