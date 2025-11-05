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

## 🖥️ Usando a Interface Gráfica

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

### Dicas para Testar:

- Use imagens de dígitos manuscritos
- Pode ser uma foto ou desenho digital
- Imagens com fundo claro e dígito escuro funcionam melhor
- Teste dígitos diferentes (0-9)

---

## 📊 Resultados Esperados

### Acurácia Típica:

- **Sem ruído:** ~96-98%
- **Com ruído (σ=2.0):** ~92-95%
- **Degradação:** ~3-5%

### Melhores Configurações:

| Parâmetro | Valor Recomendado |
|-----------|-------------------|
| Neurônios | 64-128 |
| Camadas | 1-2 camadas ocultas |
| Taxa de Aprendizado | 0.01 |
| Função de Ativação | ReLU |
| Iterações | 500 |

---

## 📸 Prints para o Relatório

### Tire screenshots de:

1. **Execução do programa principal**
   - Saída no terminal mostrando métricas

2. **Interface gráfica**
   - Imagem carregada sem ruído com predição
   - Imagem com ruído aplicado e predição
   - Diferentes dígitos sendo testados

3. **Gráficos gerados**
   - Todos os 6 gráficos salvos em PNG

---

## 📝 Estrutura do Relatório PDF

Seu relatório deve conter:

### 1. Capa
- Nome completo
- Título: "Reconhecimento de Dígitos com MLP"
- Data

### 2. Introdução
- Descrição da atividade
- Objetivos do trabalho
- Dataset utilizado (digits do scikit-learn)

### 3. Fundamentação Teórica
- O que é um Perceptron?
- O que é MLP?
- Função de ativação ReLU
- Taxa de aprendizado e bias
- Impacto do ruído em redes neurais

### 4. Metodologia
- Divisão treino/teste (70%/30%)
- Configurações testadas
- Parâmetros utilizados
- Método de inserção de ruído

### 5. Resultados
- **Tabela 1:** Variação de neurônios
- **Tabela 2:** Variação de camadas
- **Tabela 3:** Variação da taxa de aprendizado
- **Tabela 4:** Robustez ao ruído

Inclua os gráficos:
- `grafico1_neuronio.png`
- `grafico2_camadas.png`
- `grafico3_taxa_aprendizado.png`
- `grafico4_robustez_ruido.png`
- `grafico5_matriz_confusao.png`
- `grafico6_exemplos_visuais.png`

### 6. Testes com Interface Gráfica
- Screenshots da interface funcionando
- Exemplos de predições corretas
- Exemplos com ruído
- Imagens externas testadas

### 7. Respostas às Questões

**Q1: Como o ruído afetou a capacidade da rede?**
- Analise a degradação observada
- Use dados da Tabela 4 e Gráfico 4

**Q2: O aumento de camadas melhorou o desempenho?**
- Compare resultados da Tabela 2
- Discuta overfitting vs. capacidade

**Q3: A taxa de aprendizado influenciou a convergência?**
- Use dados da Tabela 3 e Gráfico 3
- Discuta velocidade vs. estabilidade

**Q4: A rede reconheceu imagens externas?**
- Mostre exemplos testados na interface
- Discuta generalização

### 8. Conclusões
- Principais descobertas
- Limitações observadas
- Aprendizados obtidos

### 9. Referências
- Documentação do scikit-learn
- Artigos sobre MLP e reconhecimento de padrões

---

## 🎯 Desafio Extra (Opcional)

Se você implementou o desafio extra, adicione ao relatório:

- Funcionalidade de aplicar ruído pela interface
- Comparação visual antes/depois do ruído
- Análise do impacto visual vs. impacto na predição

---

## ⚠️ Problemas Comuns e Soluções

### Erro: `ModuleNotFoundError: No module named 'tkinter'`
**Solução:**
- Linux: `sudo apt-get install python3-tk`
- macOS: tkinter vem com Python
- Windows: reinstale Python marcando "tcl/tk"

### Erro: Imagem não carrega
**Solução:**
- Verifique formato do arquivo (use PNG ou JPG)
- Tente redimensionar a imagem antes

### Interface não abre
**Solução:**
- Verifique se está usando ambiente gráfico (não SSH)
- Teste com: `python -m tkinter` (deve abrir janela)

### Acurácia muito baixa
**Solução:**
- Aumente número de iterações (max_iter)
- Ajuste taxa de aprendizado
- Verifique se dados foram carregados corretamente

---

## 📦 Como Compactar para Entrega

### No Windows:
1. Selecione todos os arquivos .py
2. Clique direito → "Enviar para" → "Pasta compactada"

### No Linux/Mac:
```bash
zip -r atividade_mlp.zip *.py *.png *.txt README.md
```

### Arquivos a incluir no ZIP:
- ✅ `mlp_digit_recognition.py`
- ✅ `mlp_report_generator.py`
- ✅ `README.md`
- ✅ Todos os gráficos PNG gerados (opcional)
- ✅ `relatorio_completo.txt` (opcional)

---

## 🎓 Dicas para um Relatório Excelente

1. **Seja claro e objetivo** - Explique cada experimento
2. **Use visualizações** - Gráficos ajudam a entender os resultados
3. **Analise criticamente** - Não apenas apresente dados, interprete-os
4. **Compare configurações** - Mostre qual funcionou melhor e por quê
5. **Documente tudo** - Inclua prints da execução e interface
6. **Revise** - Verifique ortografia e formatação

---

## 📞 Suporte

Se encontrar problemas:

1. Verifique se todas as bibliotecas estão instaladas
2. Confira a versão do Python (recomendado: 3.8+)
3. Leia as mensagens de erro com atenção
4. Teste cada script separadamente

---

## ✅ Checklist Final

Antes de entregar, verifique:

- [ ] Código executa sem erros
- [ ] Interface gráfica abre e funciona
- [ ] Todos os gráficos foram gerados
- [ ] Relatório PDF está completo
- [ ] Prints da interface incluídos
- [ ] Todas as questões respondidas
- [ ] Arquivos compactados em .zip
- [ ] Nome completo no relatório

---

## 🌟 Bom Trabalho!

Este projeto demonstra conceitos fundamentais de Machine Learning e Redes Neurais. Aproveite para experimentar diferentes configurações e entender como cada parâmetro afeta o desempenho!

**Desenvolvido para fins educacionais - Atividade de Redes Neurais**
