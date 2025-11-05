# 🔢 Reconhecimento de Dígitos com MLP - Instruções Completas

## 📋 Descrição do Projeto

Este projeto implementa uma Rede Neural Perceptron Multicamadas (MLP) para reconhecimento de dígitos manuscritos (0-9) com:
- ✅ Treinamento e avaliação com múltiplas configurações
- ✅ Testes de robustez com ruído artificial
- ✅ Interface gráfica interativa para testar imagens
- ✅ Geração automática de relatório completo com gráficos

---

## 🔧 Requisitos

### Bibliotecas Python necessárias:

```bash
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install seaborn
pip install pillow
pip install tkinter  # geralmente já vem com Python
```

Ou instale todas de uma vez:

```bash
pip install numpy matplotlib scikit-learn seaborn pillow
```

---

## 📁 Estrutura dos Arquivos

Você deve ter **3 arquivos Python**:

1. **`mlp_digit_recognition.py`** - Programa principal com interface gráfica
2. **`mlp_report_generator.py`** - Gerador de relatório e visualizações
3. **`README.md`** - Este arquivo de instruções

---

## 🚀 Como Executar

### Passo 1: Executar o Gerador de Relatório

Este script gera todas as visualizações e métricas necessárias:

```bash
python mlp_report_generator.py
```

**Saída esperada:**
- `grafico1_neuronio.png`
- `grafico2_camadas.png`
- `grafico3_taxa_aprendizado.png`
- `grafico4_robustez_ruido.png`
- `grafico5_matriz_confusao.png`
- `grafico6_exemplos_visuais.png`
- `relatorio_completo.txt`

### Passo 2: Executar o Programa Principal

Este script treina o modelo e abre a interface gráfica:

```bash
python mlp_digit_recognition.py
```

**O que acontece:**
1. Carrega o dataset de dígitos
2. Divide em treino/teste (70%/30%)
3. Treina o MLP com configuração padrão
4. Avalia acurácia com e sem ruído
5. Testa 8 configurações diferentes
6. Gera visualizações de exemplos
7. **Abre interface gráfica para testar imagens**

---

### Funcionalidades:

1. **📂 Carregar Imagem**
   - Clique no botão "Carregar Imagem"
   - Selecione uma imagem de dígito (PNG, JPG, etc.)
   - A imagem será automaticamente convertida para 8x8 pixels em tons de cinza

2. **🔮 Prever Dígito**
   - Após carregar, clique em "Prever Dígito"
   - Veja a predição e a confiança do modelo

3. **🔊 Adicionar Ruído**
   - Adiciona ruído gaussiano à imagem carregada
   - Visualize a imagem distorcida no segundo painel

4. **🔮 Prever com Ruído**
   - Testa o modelo na imagem ruidosa
   - Compare o resultado com a imagem original

## 📊 Resultados Esperados

### Acurácia Típica:

- **Sem ruído:** ~96-98%
- **Com ruído (σ=2.0):** ~92-95%
- **Degradação:** ~3-5%

---

## 📞 Suporte

Se encontrar problemas:

1. Verifique se todas as bibliotecas estão instaladas
2. Confira a versão do Python (recomendado: 3.8+)
3. Leia as mensagens de erro com atenção
4. Teste cada script separadamente

---

**Desenvolvido para fins educacionais - Atividade de Redes Neurais**
