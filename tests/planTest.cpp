#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Qt/PainterOstream.h>
#include <CGAL/Qt/GraphicsViewNavigation.h>

#include <QApplication>
#include <QGraphicsView>
#include <QGraphicsScene>
#include <QGraphicsItem>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point;

class TriangleItem : public QGraphicsItem
{
public:
    Point a, b, c;

    QRectF boundingRect() const override
    {
        return QRectF(-200, -200, 400, 400);
    }

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *, QWidget *) override
    {
        QRectF clip = boundingRect();
        CGAL::Qt::PainterOstream<K> out(painter, clip);
        painter->setPen(Qt::black);

        out << a << b << c;                  // ����
        out << CGAL::Triangle_2<K>(a, b, c); // ����������
    }
};

int main(int argc, char **argv)
{
    QApplication app(argc, argv);

    auto *item = new TriangleItem();
    item->a = Point(0, 0);
    item->b = Point(150, 0);
    item->c = Point(50, 120);

    QGraphicsScene scene;
    scene.addItem(item);

    QGraphicsView view(&scene);

    CGAL::Qt::GraphicsViewNavigation navigation;
    view.setRenderHint(QPainter::Antialiasing);

    view.resize(600, 600);
    view.show();

    return app.exec();
}
