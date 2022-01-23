#include <chrono>
#include <iostream>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <dlib/clustering.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>

using namespace std;
using namespace dlib;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;

int main(int argc, char const *argv[]) {
    try {
        auto tempo_total_inicio = std::chrono::high_resolution_clock::now();

        // Carrega o detector hog
        frontal_face_detector detectorHog = get_frontal_face_detector();
        
        // Carrega o shape predictor
        shape_predictor shapePredictor;
        deserialize("shape_predictor_68_face_landmarks.dat") >> shapePredictor;
        
        // Caminho da imagem 1
        string caminho_imagem_1 = "pessoas/11-12.jpg";

        // Caminho da imagem 2
        string caminho_imagem_2 = "pessoas/26-12.jpg";

        // Pega o caminho das imagens pelo argumento
        if (argc == 3) {
            caminho_imagem_1 = argv[1];
            caminho_imagem_2 = argv[2];
        }

        // Carrega a imagem 1
        array2d<rgb_pixel> imagem_1;
        load_image(imagem_1, caminho_imagem_1);
        // Converte a imagem 1 para gray
        array2d<unsigned char> imagem_cinza_1;
        assign_image(imagem_cinza_1, imagem_1);
        // Detecta as faces na imagem 1
        std::vector<rectangle> faces_detectadas_hog_1 = detectorHog(imagem_cinza_1);
        int size_1 = faces_detectadas_hog_1.size();
        if (size_1 == 0) {
            cout << "Nenhuma Face Detectada na Imagem 1" << endl;
            return 1;
        }
        // Pega a primeira face detectada
        rectangle face_1 = faces_detectadas_hog_1[0];
        
        // Carrega a imagem 2
        array2d<rgb_pixel> imagem_2;
        load_image(imagem_2, caminho_imagem_2);
        // Converte a imagem 2 para gray
        array2d<unsigned char> imagem_cinza_2;
        assign_image(imagem_cinza_2, imagem_2);
        // Detecta as faces na imagem 2
        std::vector<rectangle> faces_detectadas_hog_2 = detectorHog(imagem_cinza_2);
        int size_2 = faces_detectadas_hog_2.size();
        if (size_2 == 0) {
            cout << "Nenhuma Face Detectada na Imagem 2" << endl;
            return 1;
        }
        // Pega a primeira face detectada
        rectangle face_2 = faces_detectadas_hog_2[0];
        
        // detectando os pontos faciais da imagem 1
        full_object_detection shape_1 = shapePredictor(imagem_1, face_1);
        
        // detectando os pontos faciais da imagem 2
        full_object_detection shape_2 = shapePredictor(imagem_2, face_2);
        
        // pego apena as faces recortadas das pessoas para fazer o treinamento
        std::vector<matrix<rgb_pixel>> faces;
        
        // recorto a face 1
        matrix<rgb_pixel> face_chip_1;
        extract_image_chip(imagem_1, get_face_chip_details(shape_1, 150, 0.25), face_chip_1);
        faces.push_back(move(face_chip_1));
        
        // recorto a face 2
        matrix<rgb_pixel> face_chip_2;
        extract_image_chip(imagem_2, get_face_chip_details(shape_2, 150, 0.25), face_chip_2);
        faces.push_back(move(face_chip_2));
        
        // Carrega o classificador
        anet_type net;
        deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
        
        // Fazendo a predição
        auto tempo_inicio_predicao = std::chrono::high_resolution_clock::now();
        std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
        auto tempo_fim_predicao = std::chrono::high_resolution_clock::now();

        if (face_descriptors.size() == 2) {
            cout << "Distancia: " << length(face_descriptors[0] - face_descriptors[1]) << endl;
        }

        // Mostra o Tempo Total
        auto tempo_total_fim = std::chrono::high_resolution_clock::now();
        double tempo_total = std::chrono::duration<double, std::milli>(tempo_total_fim - tempo_total_inicio).count();
        cout << "Tempo Total: " << to_string(tempo_total) << endl;

        // Mostra o Tempo de Predição
        double tempo_predicao = std::chrono::duration<double, std::milli>(tempo_fim_predicao - tempo_inicio_predicao).count();
        cout << "Tempo Predicao: " << to_string(tempo_predicao) << endl;

        // Mostra a Imagem 1 em uma Janela
        image_window my_window_1(imagem_1, "Imagem 1");
        my_window_1.add_overlay(faces_detectadas_hog_1, rgb_pixel(0, 255, 0));
        my_window_1.add_overlay(render_face_detections(shape_1));
        // my_window_1.wait_until_closed();

        // Mostra a Imagem 2 em uma Janela
        image_window my_window_2(imagem_2, "Imagem 2");
        my_window_2.add_overlay(faces_detectadas_hog_2, rgb_pixel(0, 255, 0));
        my_window_2.add_overlay(render_face_detections(shape_2));
        my_window_2.wait_until_closed();

    } catch (exception& e) {
        cout << "\nexception thrown!" << endl;
        cout << e.what() << endl;
        return 1;
    }
}