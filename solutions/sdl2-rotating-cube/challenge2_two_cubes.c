/*
 * Challenge 2: Two Cubes Side by Side
 *
 * Extends step4_cube.c — draws two cubes with different rotation speeds,
 * offset horizontally. Uses the same VBO/IBO, two different MVP matrices.
 *
 * Build:  cmake --build build --target challenge2_two_cubes
 * Run:    SDL_VIDEODRIVER=kmsdrm ./build/challenge2_two_cubes
 *
 * Key changes from step4_cube.c:
 *   - Two draw calls per frame, each with a different MVP
 *   - mat4_translate offsets each cube left/right
 *   - Different rotation speeds show they're independent
 */

#include <SDL2/SDL.h>
#include <GLES2/gl2.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

/* ── Shader helpers (same as step4) ──────────────────── */

static GLuint compile_shader(GLenum type, const char *src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), NULL, log);
        fprintf(stderr, "Shader error: %s\n", log);
        return 0;
    }
    return s;
}

static GLuint link_program(GLuint vs, GLuint fs)
{
    GLuint p = glCreateProgram();
    glAttachShader(p, vs);
    glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(p, sizeof(log), NULL, log);
        fprintf(stderr, "Link error: %s\n", log);
        return 0;
    }
    return p;
}

/* ── Matrix math (same as step4) ─────────────────────── */

static void mat4_identity(float m[16])
{
    memset(m, 0, 16 * sizeof(float));
    m[0] = m[5] = m[10] = m[15] = 1.0f;
}

static void mat4_mul(float out[16], const float a[16], const float b[16])
{
    float r[16];
    for (int c = 0; c < 4; c++)
        for (int row = 0; row < 4; row++)
            r[c * 4 + row] = a[0 * 4 + row] * b[c * 4 + 0]
                            + a[1 * 4 + row] * b[c * 4 + 1]
                            + a[2 * 4 + row] * b[c * 4 + 2]
                            + a[3 * 4 + row] * b[c * 4 + 3];
    memcpy(out, r, sizeof(r));
}

static void mat4_perspective(float m[16], float fovy_rad,
                             float aspect, float znear, float zfar)
{
    float f = 1.0f / tanf(fovy_rad * 0.5f);
    memset(m, 0, 16 * sizeof(float));
    m[0]  = f / aspect;
    m[5]  = f;
    m[10] = (zfar + znear) / (znear - zfar);
    m[11] = -1.0f;
    m[14] = (2.0f * zfar * znear) / (znear - zfar);
}

static void mat4_translate(float m[16], float x, float y, float z)
{
    mat4_identity(m);
    m[12] = x; m[13] = y; m[14] = z;
}

static void mat4_rotate_y(float m[16], float a)
{
    mat4_identity(m);
    m[0] = cosf(a);  m[8]  = sinf(a);
    m[2] = -sinf(a); m[10] = cosf(a);
}

static void mat4_rotate_x(float m[16], float a)
{
    mat4_identity(m);
    m[5] = cosf(a);  m[9]  = -sinf(a);
    m[6] = sinf(a);  m[10] = cosf(a);
}

/* ── Draw one cube with a given MVP matrix ───────────── */
static void draw_cube(GLint loc_mvp, const float mvp[16],
                      GLint loc_pos, GLint loc_col)
{
    glUniformMatrix4fv(loc_mvp, 1, GL_FALSE, mvp);

    glEnableVertexAttribArray((GLuint)loc_pos);
    glVertexAttribPointer((GLuint)loc_pos, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void *)0);
    glEnableVertexAttribArray((GLuint)loc_col);
    glVertexAttribPointer((GLuint)loc_col, 3, GL_FLOAT, GL_FALSE,
                          6 * sizeof(float), (void *)(3 * sizeof(float)));

    glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_SHORT, (void *)0);
}

/* ── Main ───────────────────────────────────────────── */

int main(int argc, char **argv)
{
    (void)argc; (void)argv;

    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    int w = 800, h = 480;
    SDL_Window *win = SDL_CreateWindow("Challenge 2: Two Cubes",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
        w, h, SDL_WINDOW_OPENGL | SDL_WINDOW_FULLSCREEN_DESKTOP);
    if (!win) {
        fprintf(stderr, "Window: %s\n", SDL_GetError());
        SDL_Quit(); return 1;
    }

    SDL_GLContext ctx = SDL_GL_CreateContext(win);
    if (!ctx) {
        fprintf(stderr, "GL context: %s\n", SDL_GetError());
        SDL_DestroyWindow(win); SDL_Quit(); return 1;
    }
    SDL_GL_SetSwapInterval(1);

    printf("GL Renderer: %s\n", glGetString(GL_RENDERER));

    /* ── Shaders ── */
    const char *vs_src =
        "attribute vec3 aPos;\n"
        "attribute vec3 aCol;\n"
        "uniform mat4 uMVP;\n"
        "varying vec3 vCol;\n"
        "void main() {\n"
        "    vCol = aCol;\n"
        "    gl_Position = uMVP * vec4(aPos, 1.0);\n"
        "}\n";

    const char *fs_src =
        "precision mediump float;\n"
        "varying vec3 vCol;\n"
        "void main() {\n"
        "    gl_FragColor = vec4(vCol, 1.0);\n"
        "}\n";

    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    GLuint prog = link_program(vs, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    if (!prog) return 1;

    GLint loc_pos = glGetAttribLocation(prog, "aPos");
    GLint loc_col = glGetAttribLocation(prog, "aCol");
    GLint loc_mvp = glGetUniformLocation(prog, "uMVP");

    /* ── Cube geometry (shared between both cubes) ── */
    const float verts[] = {
        -1.f, -1.f, -1.f,   1.f, 0.f, 0.f,
         1.f, -1.f, -1.f,   0.f, 1.f, 0.f,
         1.f,  1.f, -1.f,   0.f, 0.f, 1.f,
        -1.f,  1.f, -1.f,   1.f, 1.f, 0.f,
        -1.f, -1.f,  1.f,   1.f, 0.f, 1.f,
         1.f, -1.f,  1.f,   0.f, 1.f, 1.f,
         1.f,  1.f,  1.f,   1.f, 1.f, 1.f,
        -1.f,  1.f,  1.f,   0.3f, 0.3f, 0.3f,
    };

    const GLushort indices[] = {
        0,1,2, 2,3,0,  4,5,6, 6,7,4,
        0,4,7, 7,3,0,  1,5,6, 6,2,1,
        3,2,6, 6,7,3,  0,1,5, 5,4,0,
    };

    GLuint vbo, ibo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);

    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glEnable(GL_DEPTH_TEST);

    /* ── Render loop ── */
    Uint64 t_start = SDL_GetPerformanceCounter();
    double freq = (double)SDL_GetPerformanceFrequency();
    Uint64 t_fps = t_start;
    int frames = 0;

    int running = 1;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = 0;
            if (e.type == SDL_KEYDOWN &&
                (e.key.keysym.sym == SDLK_ESCAPE ||
                 e.key.keysym.sym == SDLK_q)) running = 0;
            if (e.type == SDL_WINDOWEVENT &&
                e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                w = e.window.data1;
                h = e.window.data2;
            }
        }

        double sec = (double)(SDL_GetPerformanceCounter() - t_start) / freq;

        glViewport(0, 0, w, h);
        glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(prog);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

        float P[16];
        mat4_perspective(P, 60.0f * (3.14159f / 180.0f),
                         (float)w / (float)h, 0.1f, 100.0f);

        /* ── CUBE 1: left side, fast rotation ── */
        {
            float T[16], Rx[16], Ry[16], Rxy[16], M[16], MVP[16];
            mat4_translate(T, -2.0f, 0.f, -6.0f);  /* offset left */
            mat4_rotate_y(Ry, (float)sec * 1.5f);   /* fast Y */
            mat4_rotate_x(Rx, (float)sec * 0.7f);
            mat4_mul(Rxy, Ry, Rx);
            mat4_mul(M, T, Rxy);
            mat4_mul(MVP, P, M);
            draw_cube(loc_mvp, MVP, loc_pos, loc_col);
        }

        /* ── CUBE 2: right side, slow rotation ── */
        {
            float T[16], Rx[16], Ry[16], Rxy[16], M[16], MVP[16];
            mat4_translate(T, 2.0f, 0.f, -6.0f);   /* offset right */
            mat4_rotate_y(Ry, (float)sec * 0.4f);   /* slow Y */
            mat4_rotate_x(Rx, (float)sec * 1.2f);   /* fast X */
            mat4_mul(Rxy, Ry, Rx);
            mat4_mul(M, T, Rxy);
            mat4_mul(MVP, P, M);
            draw_cube(loc_mvp, MVP, loc_pos, loc_col);
        }

        SDL_GL_SwapWindow(win);

        /* FPS reporting */
        frames++;
        Uint64 now = SDL_GetPerformanceCounter();
        double dt = (double)(now - t_fps) / freq;
        if (dt >= 2.0) {
            printf("FPS: %.1f  (2 cubes, 2 draw calls)\n", frames / dt);
            frames = 0;
            t_fps = now;
        }
    }

    glDeleteProgram(prog);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);
    SDL_GL_DeleteContext(ctx);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return 0;
}
