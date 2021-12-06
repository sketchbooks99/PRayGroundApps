#include "app.h"

int main()
{
    pgSetAppDir(APP_DIR);

    auto window = std::make_shared<Window>("Cornel", 1024, 1024);
    auto app = std::make_shared<App>();

    pgRunApp(app, window);
}