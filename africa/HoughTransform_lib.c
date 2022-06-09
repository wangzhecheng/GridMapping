#include <math.h>
#include <stdlib.h>
#define M_PI 3.14159265358979323846
#define IMGSIZE 77
#define CAMSIZE (IMGSIZE * IMGSIZE)
#define THETARANGE 180
#define LINEMAX 10
#define maskDepth 9999
#define maskWidth 3

#ifdef _MSC_VER
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

double *cos_t, *sin_t;
double *thetas, *rhos;
int* accumulator;
int theta_len, diag_len;
int weight, height;
double* cam;
int* cam_regular;
double* lines;
int lines_len;

struct Points {
    int x1, y1, x2, y2;
};

struct Points polar2cart(double rho, double theta) {
    double a = cos(theta);
    double b = sin(theta);
    double x0 = a * rho;
    double y0 = b * rho;
    struct Points p;
    p.x1 = (int)(x0 + 1000 * (-b));
    p.y1 = (int)(y0 + 1000 * (a));
    p.x2 = (int)(x0 - 1000 * (-b));
    p.y2 = (int)(y0 - 1000 * (a));
    return p;
}

int getMaxRank(int* array, int len) {
    int rank = 0;
    int maxValue = array[0];
    for (int i = 1; i < len; i++) {
        if (array[i] > maxValue) {
            maxValue = array[i];
            rank = i;
        }
    }
    return rank;
}

int getMax(int* array, int len) {
    return array[getMaxRank(array, len)];
}

void CAMRegular() {
    int level1 = 10;
    int level2 = 1;
    int level1_thre = 80;
    int level2_thre = 10;
    for (int i = 0; i < CAMSIZE; i++) {
        if (cam[i] > level1_thre)
            cam_regular[i] = level1;
        else if (cam[i] > level2_thre)
            cam_regular[i] = level2;
        else
            cam_regular[i] = 0;
    }
}

int capsule(double px,
            double py,
            double ax,
            double ay,
            double bx,
            double by,
            double r) {
    double pax = px - ax, pay = py - ay, bax = bx - ax, bay = by - ay;
    double h = fmaxf(
        fminf((pax * bax + pay * bay) / (bax * bax + bay * bay), 1.0f), 0.0f);
    double dx = pax - bax * h, dy = pay - bay * h;
    return dx * dx + dy * dy < r * r;
}

void lineMask(double* img, struct Points pts, int width) {
    double* p = img;
    for (int y = 0; y < IMGSIZE; y++)
        for (int x = 0; x < IMGSIZE; x++, p += 1)
            *p = *p - capsule(x, y, pts.x1, pts.y1, pts.x2, pts.y2, width) *
                          maskDepth;
    return;
}

DLL_EXPORT void init() {
    theta_len = THETARANGE + 1;
    weight = IMGSIZE;
    height = IMGSIZE;
    diag_len = (int)(ceil(sqrt(weight * weight + height * height)));
    thetas = (double*)malloc(sizeof(double) * theta_len);
    cos_t = (double*)malloc(sizeof(double) * theta_len);
    sin_t = (double*)malloc(sizeof(double) * theta_len);
    rhos = (double*)malloc(sizeof(double) * 2 * diag_len);
    accumulator = (int*)malloc(sizeof(int) * 2 * diag_len * theta_len);
    cam = (double*)malloc(sizeof(double) * CAMSIZE);
    cam_regular = (int*)malloc(sizeof(int) * CAMSIZE);
    lines = (double*)malloc(sizeof(double) * 3 * LINEMAX);
    for (int i = 0; i < theta_len; i++) {
        thetas[i] = (i - 90) * M_PI / 180;
        cos_t[i] = cos(thetas[i]);
        sin_t[i] = sin(thetas[i]);
    }
    double rhos_delta = 2 * diag_len / (2 * diag_len - 1);
    for (int i = 0; i < 2 * diag_len; i++)
        rhos[i] = -diag_len + i * rhos_delta;
}

DLL_EXPORT int* HoughTransform(int* img) {
    for (int i = 0; i < 2 * diag_len; i++)
        for (int j = 0; j < theta_len; j++)
            accumulator[i * theta_len + j] = 0;

    for (int x = 0; x < weight; x++)
        for (int y = 0; y < height; y++) {
            if (img[y * weight + x] == 0)
                return;
            for (int t = 0; t < theta_len; t++) {
                int rho = (int)(round(x * cos_t[t] + y * sin_t[t])) + diag_len;
                accumulator[rho * theta_len + t] += img[y * weight + x];
            }
        }
    return accumulator;
}
DLL_EXPORT double* getth() {
    return thetas;
}
DLL_EXPORT double* getrh() {
    return rhos;
}
DLL_EXPORT int getthlen() {
    return theta_len;
}
DLL_EXPORT int getrhlen() {
    return 2 * diag_len;
}
DLL_EXPORT void destory() {
    free(thetas);
    free(sin_t);
    free(cos_t);
    free(rhos);
    free(accumulator);
    free(cam);
    free(cam_regular);
    free(lines);
}

DLL_EXPORT int cam2lines(double* cam_data) {
    for (int i = 0; i < CAMSIZE; i++)
        cam[i] = cam_data[i];
    // memcpy(cam, cam_data, sizeof(double) * CAMSIZE);
    lines_len = 0;
    int line_thre = 80;
    for (int i = 0; i < LINEMAX; i++) {
        CAMRegular();
        int* hf = HoughTransform(cam_regular);
        if (getMax(hf, 2 * diag_len * theta_len) < line_thre)
            break;
        int hfLine = getMaxRank(hf, 2 * diag_len * theta_len);
        int hfLine_r = hfLine / theta_len;
        int hfLine_t = hfLine % theta_len;
        double rho = rhos[hfLine_r];
        double theta = thetas[hfLine_t];
        lines[i * 3 + 0] = rho;
        lines[i * 3 + 1] = theta;
        lines[i * 3 + 2] = hf[hfLine];
        lines_len++;
        struct Points p = polar2cart(rho, theta);
        lineMask(cam, p, maskWidth);
    }
    return lines_len;
}

DLL_EXPORT double* getlines() {
    return lines;
}
